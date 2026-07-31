"""
_segmentation_backend.py

Generación de máscaras para COLMAP.

CONVENCIÓN (la que espera COLMAP con --ImageReader.mask_path):
    BLANCO (255) = el píxel SÍ se procesa.
    NEGRO  (0)   = el píxel se ignora.

Esto es justo lo contrario de lo que hacía la versión anterior de este
fichero, que guardaba un "indexed_mask" con los IDs de clase en crudo
(0 = fondo, 1..7 = categorías) directamente como PNG en escala de grises.
Como el fondo era la clase 0, la imagen entera salía negra, con manchas
grises casi invisibles (valores 1-7 sobre 255) para las regiones detectadas.
Es decir: estaba invertida Y mal escalada a la vez.

Filosofía de esta reescritura:

- Las categorías "estáticas" de bajo riesgo para el matching (cielo, agua,
  reflejos) se pueden excluir automáticamente en modo desatendido, porque
  son casi universalmente problemáticas en fotogrametría. Se pueden
  desactivar con --no-auto-environment-mask.
- Las categorías dependientes del contexto (suelo, vegetación, paredes
  lisas / fondo desenfocado) SOLO se excluyen si el usuario las pide
  explícitamente en modo atendido. Nunca se infieren solas.
- Las categorías "dinámicas" (personas, vehículos, animales, aves) SOLO se
  excluyen si:
    1) el usuario las pide explícitamente en modo atendido, o
    2) en modo automático, el analizador de secuencia (_scene_analyzer.py)
       las marca como anomalía respecto al resto de la escena (aparecen de
       forma transitoria o se mueven de forma inconsistente con el
       movimiento de cámara). Nunca se enmascara "toda persona/coche" solo
       por pertenecer a esa clase.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categorías y palabras clave (ES/EN) para el modo atendido
# ---------------------------------------------------------------------------

CAT_SKY = "sky"
CAT_WATER = "water"
CAT_REFLECTION = "reflection"
CAT_GROUND = "ground"
CAT_VEGETATION = "vegetation"
CAT_UNIFORM_WALL = "uniform_wall"      # también cubre "fondo desenfocado"
CAT_PERSON = "person"
CAT_VEHICLE = "vehicle"
CAT_ANIMAL = "animal"
CAT_BIRD = "bird"

STATIC_CATEGORIES = {CAT_SKY, CAT_WATER, CAT_REFLECTION, CAT_GROUND, CAT_VEGETATION, CAT_UNIFORM_WALL}
DYNAMIC_CATEGORIES = {CAT_PERSON, CAT_VEHICLE, CAT_ANIMAL, CAT_BIRD}

# Categorías estáticas que se excluyen por defecto en modo automático
# (universalmente problemáticas para el matching de features).
AUTO_ENVIRONMENT_CATEGORIES = {CAT_SKY, CAT_WATER, CAT_REFLECTION}

CATEGORY_LABELS_ES = {
    CAT_SKY: "cielo",
    CAT_WATER: "agua",
    CAT_REFLECTION: "reflejos / sobreexposición",
    CAT_GROUND: "suelo",
    CAT_VEGETATION: "vegetación",
    CAT_UNIFORM_WALL: "superficies lisas / fondo desenfocado",
    CAT_PERSON: "personas",
    CAT_VEHICLE: "vehículos",
    CAT_ANIMAL: "animales",
    CAT_BIRD: "aves",
}

_CATEGORY_KEYWORDS = {
    CAT_SKY: ["cielo", "sky", "nubes", "clouds"],
    CAT_WATER: ["agua", "water", "mar", "rio", "río", "lago", "charco"],
    CAT_REFLECTION: ["reflejo", "reflejos", "reflection", "reflections", "brillo", "brillos", "cristal", "cristales", "glass", "sobreexpos"],
    CAT_GROUND: ["suelo", "piso", "ground", "floor", "terreno", "pavimento", "asfalto"],
    CAT_VEGETATION: ["vegetacion", "vegetación", "plantas", "vegetation", "arbol", "árbol", "arboles", "árboles", "hierba", "cesped", "césped", "hojas"],
    CAT_UNIFORM_WALL: ["pared", "paredes", "muro", "muros", "wall", "walls", "liso", "lisa", "uniform", "desenfoque", "borroso", "borrosa", "blur", "bokeh", "fondo desenfocado", "fondo borroso"],
    CAT_PERSON: ["persona", "personas", "gente", "people", "person", "humano", "humanos", "peaton", "peatón", "peatones"],
    CAT_VEHICLE: ["coche", "coches", "vehiculo", "vehículo", "vehiculos", "vehículos", "car", "cars", "vehicle", "vehicles", "trafico", "tráfico", "auto", "autos", "moto", "motos", "camion", "camión", "bici", "bicicleta"],
    CAT_ANIMAL: ["animal", "animales", "animals", "mascota", "mascotas", "perro", "perros", "gato", "gatos"],
    CAT_BIRD: ["pajaro", "pájaro", "pajaros", "pájaros", "ave", "aves", "bird", "birds"],
}


def parse_ignore_request(text: str) -> Set[str]:
    """
    Convierte la respuesta libre del usuario a "What do you want to ignore
    from images?" en un conjunto de categorías reconocidas.

    Es intencionadamente tolerante: coincidencia de subcadena, sin acentos
    estrictos, admite listas separadas por comas, 'y', '/', etc.
    """
    if not text:
        return set()

    normalized = text.lower()
    found: Set[str] = set()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                found.add(category)
                break
    return found


def prompt_ignore_categories(input_fn=input) -> Set[str]:
    """
    Pregunta interactivamente qué categorías ignorar. Se llama UNA vez por
    ejecución (no por frame): la respuesta se aplica a toda la secuencia.
    `input_fn` es inyectable para tests.
    """
    print(
        "\nModo atendido — máscaras.\n"
        "What do you want to ignore from images? "
        "(ej: cielo, suelo, personas, vehiculos, vegetacion, agua, reflejos, "
        "paredes lisas / fondo desenfocado, animales, aves)\n"
        "Deja vacío para no excluir nada."
    )
    raw = input_fn("> ").strip()
    categories = parse_ignore_request(raw)

    if categories:
        labels = ", ".join(CATEGORY_LABELS_ES[c] for c in sorted(categories))
        print(f"Se ignorará: {labels}")
    else:
        print("No se excluirá ninguna categoría (máscaras en blanco / sin recorte).")

    return categories


# ---------------------------------------------------------------------------
# YOLO (personas / vehículos / animales / aves) — usado tanto en modo
# atendido (si el usuario pide esas categorías) como por el analizador de
# escena en modo automático (_scene_analyzer.py llama a estas mismas
# funciones para no duplicar lógica).
# ---------------------------------------------------------------------------

# Clases COCO relevantes.
YOLO_PERSON_CLASSES = {0}
YOLO_VEHICLE_CLASSES = {1, 2, 3, 5, 6, 7}   # bicycle, car, motorcycle, bus, train, truck
YOLO_BIRD_CLASSES = {14}
YOLO_ANIMAL_CLASSES = {15, 16, 17, 18, 19, 20, 21, 22, 23}  # cat..giraffe

_YOLO_MODEL: Optional[Any] = None
_YOLO_UNAVAILABLE_REASON: Optional[str] = None


def _checkpoints_dir() -> Path:
    """
    Carpeta de checkpoints en la raíz del proyecto (no en el cwd desde donde
    se lance el comando). Se crea si no existe.
    """
    # .../src/phases/_segmentation_backend.py -> raíz del proyecto = parents[2]
    project_root = Path(__file__).resolve().parents[2]
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_yolo_model(weights_name: str = "yolov8n.pt") -> Optional[Any]:
    """
    Carga (con caché a nivel de módulo) el modelo YOLO usado para detectar
    personas/vehículos/animales/aves.

    El checkpoint se guarda/busca en <raíz_del_proyecto>/checkpoints/, no en
    la carpeta desde la que se ejecuta el script (bug anterior: al pasar
    solo "yolov8n.pt", ultralytics lo descargaba en el cwd).
    """
    global _YOLO_MODEL, _YOLO_UNAVAILABLE_REASON

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if _YOLO_UNAVAILABLE_REASON is not None:
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:
        _YOLO_UNAVAILABLE_REASON = f"ultralytics no disponible: {exc}"
        logger.warning(
            "%s. No se detectarán personas/vehículos/animales/aves "
            "(instala 'ultralytics' para activarlo).",
            _YOLO_UNAVAILABLE_REASON,
        )
        return None

    ckpt_path = _checkpoints_dir() / weights_name

    try:
        # Si el checkpoint ya existe en checkpoints/, se usa directamente.
        # Si no, se le pasa la ruta completa de destino para que ultralytics
        # lo descargue ahí (en vez de en el cwd).
        _YOLO_MODEL = YOLO(str(ckpt_path) if ckpt_path.exists() else weights_name)
        if not ckpt_path.exists():
            # ultralytics descarga con el nombre simple en el cwd; lo
            # movemos a checkpoints/ para que quede donde corresponde.
            downloaded = Path(weights_name)
            if downloaded.exists():
                downloaded.replace(ckpt_path)
                logger.info("Checkpoint YOLO movido a %s", ckpt_path)
    except Exception as exc:
        _YOLO_UNAVAILABLE_REASON = f"No se pudo cargar el modelo YOLO '{weights_name}': {exc}"
        logger.warning("%s. No se detectarán personas/vehículos/animales/aves.", _YOLO_UNAVAILABLE_REASON)
        return None

    logger.info("Modelo YOLO '%s' cargado desde %s", weights_name, ckpt_path)
    return _YOLO_MODEL


def detect_dynamic_boxes(img_bgr: np.ndarray, conf: float = 0.35) -> List[Dict[str, Any]]:
    """
    Devuelve una lista de detecciones dinámicas en un frame:
    [{"category": "person"|"vehicle"|"animal"|"bird", "box": (x1,y1,x2,y2), "conf": float}, ...]

    Degradación controlada: si YOLO no está disponible o falla, devuelve [].
    """
    model = get_yolo_model()
    if model is None:
        return []

    h, w = img_bgr.shape[:2]
    detections: List[Dict[str, Any]] = []

    try:
        results = model.predict(source=img_bgr, verbose=False, conf=conf)
    except Exception:
        logger.warning("Fallo en la inferencia YOLO; se omite este frame.", exc_info=True)
        return []

    try:
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0]) if box.conf is not None else conf
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                if cls_id in YOLO_PERSON_CLASSES:
                    category = CAT_PERSON
                elif cls_id in YOLO_VEHICLE_CLASSES:
                    category = CAT_VEHICLE
                elif cls_id in YOLO_BIRD_CLASSES:
                    category = CAT_BIRD
                elif cls_id in YOLO_ANIMAL_CLASSES:
                    category = CAT_ANIMAL
                else:
                    continue

                detections.append({"category": category, "box": (x1, y1, x2, y2), "conf": conf_val})
    except Exception:
        logger.warning("Fallo interpretando resultados de YOLO; se omite este frame.", exc_info=True)
        return []

    return detections


# ---------------------------------------------------------------------------
# Segmentación clásica (CV) para categorías estáticas
# ---------------------------------------------------------------------------

def _cleanup_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def _low_texture_mask(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(np.abs(lap), (9, 9), 0)
    norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.inRange(norm, 0, 20)


def compute_static_masks(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcula, para todas las categorías estáticas, una máscara binaria
    (255 = pertenece a la categoría) mediante heurísticas clásicas de CV.

    Se calculan SIEMPRE (para poder informar de los porcentajes en el
    report), independientemente de si luego se usan para recortar o no.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    upper_half = np.zeros((h, w), dtype=np.uint8)
    upper_half[: h // 2, :] = 255
    sky_color = cv2.inRange(hsv, (85, 20, 80), (135, 255, 255))
    sky_mask = _cleanup_mask(cv2.bitwise_and(sky_color, upper_half), 5)

    lower_half = np.zeros((h, w), dtype=np.uint8)
    lower_half[h // 3:, :] = 255
    water_color = cv2.inRange(hsv, (90, 20, 40), (140, 255, 220))
    water_mask = _cleanup_mask(cv2.bitwise_and(water_color, lower_half), 5)

    vegetation_mask = _cleanup_mask(cv2.inRange(hsv, (30, 25, 25), (90, 255, 255)), 5)

    reflection_mask = _cleanup_mask(cv2.inRange(gray, 235, 255), 3)

    # "Suelo": franja inferior de la imagen que no es agua, con textura
    # razonablemente uniforme. Heurística simple, no una segmentación
    # semántica real: se documenta como tal en el report.
    bottom_band = np.zeros((h, w), dtype=np.uint8)
    bottom_band[int(h * 0.66):, :] = 255
    ground_mask = _cleanup_mask(cv2.bitwise_and(bottom_band, cv2.bitwise_not(water_mask)), 7)

    uniform_wall_mask = _cleanup_mask(_low_texture_mask(gray), 5)

    return {
        CAT_SKY: sky_mask,
        CAT_WATER: water_mask,
        CAT_VEGETATION: vegetation_mask,
        CAT_REFLECTION: reflection_mask,
        CAT_GROUND: ground_mask,
        CAT_UNIFORM_WALL: uniform_wall_mask,
    }


def _pct(binary_mask: np.ndarray, total_px: float) -> float:
    return round((float(np.count_nonzero(binary_mask)) / total_px) * 100.0, 3)


# ---------------------------------------------------------------------------
# Construcción de la máscara final (blanco=procesar / negro=ignorar)
# ---------------------------------------------------------------------------

def build_mask_for_frame(
    img_bgr: np.ndarray,
    requested_static_categories: Set[str],
    dynamic_boxes: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Construye la máscara final para un frame.

    - requested_static_categories: categorías estáticas a excluir para ESTE
      frame (ya decididas antes de llamar: por el usuario en modo atendido,
      o por AUTO_ENVIRONMENT_CATEGORIES en modo automático).
    - dynamic_boxes: cajas [{"category":..., "box": (x1,y1,x2,y2)}, ...] que
      ya se decidió excluir para este frame (petición explícita del usuario
      o anomalía detectada por el analizador de secuencia).

    Devuelve (keep_mask, metrics, debug_visualization).
    keep_mask: uint8 HxW, 255 = procesar, 0 = ignorar (formato COLMAP).
    """
    h, w = img_bgr.shape[:2]
    total_px = float(h * w)

    static_masks = compute_static_masks(img_bgr)

    # Máscara "a excluir" acumulada. Se parte de todo-en-blanco (todo se
    # procesa) y se va pintando de negro lo que se decide ignorar.
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    applied_static = []
    for category in requested_static_categories & STATIC_CATEGORIES:
        exclude_mask = cv2.bitwise_or(exclude_mask, static_masks[category])
        applied_static.append(category)

    for det in dynamic_boxes:
        x1, y1, x2, y2 = det["box"]
        exclude_mask[y1:y2, x1:x2] = 255

    keep_mask = cv2.bitwise_not(exclude_mask)

    # Métricas SIEMPRE informativas (se calculan aunque no se usen para
    # recortar), para que el report explique el contenido de la escena.
    metrics = {
        f"{cat}_pct": _pct(mask, total_px) for cat, mask in static_masks.items()
    }
    dynamic_px_mask = np.zeros((h, w), dtype=np.uint8)
    for det in dynamic_boxes:
        x1, y1, x2, y2 = det["box"]
        dynamic_px_mask[y1:y2, x1:x2] = 255
    metrics["dynamic_masked_pct"] = _pct(dynamic_px_mask, total_px)
    metrics["masked_categories"] = sorted(set(applied_static) | {d["category"] for d in dynamic_boxes})
    metrics["usable_area_pct"] = _pct(keep_mask, total_px)

    debug = _build_debug_visualization(img_bgr, static_masks, dynamic_boxes, applied_static)

    return keep_mask, metrics, debug


_DEBUG_COLORS_BGR = {
    CAT_SKY: (255, 128, 0),
    CAT_WATER: (255, 0, 0),
    CAT_VEGETATION: (0, 180, 0),
    CAT_REFLECTION: (0, 255, 255),
    CAT_GROUND: (0, 128, 255),
    CAT_UNIFORM_WALL: (128, 128, 128),
    CAT_PERSON: (180, 0, 180),
    CAT_VEHICLE: (0, 0, 255),
    CAT_ANIMAL: (255, 0, 255),
    CAT_BIRD: (0, 255, 128),
}


def _build_debug_visualization(
    img_bgr: np.ndarray,
    static_masks: Dict[str, np.ndarray],
    dynamic_boxes: List[Dict[str, Any]],
    applied_static: List[str],
) -> np.ndarray:
    """
    Genera una imagen de depuración semitransparente: solo colorea las
    categorías que se han excluido realmente en este frame, para que un
    humano pueda comprobar de un vistazo que la máscara tiene sentido.
    Esto es aparte de la máscara binaria real que se pasa a COLMAP.
    """
    overlay = img_bgr.copy()
    for category in applied_static:
        color = _DEBUG_COLORS_BGR[category]
        overlay[static_masks[category] > 0] = color

    for det in dynamic_boxes:
        x1, y1, x2, y2 = det["box"]
        color = _DEBUG_COLORS_BGR.get(det["category"], (0, 0, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    return cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)


# ---------------------------------------------------------------------------
# Orquestación de alto nivel usada por phase2_semantic.py
# ---------------------------------------------------------------------------

def run_segmentation(
    frame_list: Iterable[str],
    masks_dir: str | Path,
    requested_static_categories: Optional[Set[str]] = None,
    dynamic_boxes_per_frame: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    write_debug_visualizations: bool = True,
) -> List[Dict[str, Any]]:
    """
    Genera las máscaras (convención blanco=procesar/negro=ignorar) y las
    métricas por frame para toda la secuencia.

    - requested_static_categories: set de categorías estáticas a excluir en
      TODOS los frames (decisión global: pedida por el usuario en modo
      atendido, o AUTO_ENVIRONMENT_CATEGORIES en modo automático).
    - dynamic_boxes_per_frame: {nombre_de_frame: [detecciones...]} ya
      decididas de antemano (petición explícita o anomalías detectadas).
    """
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = masks_dir / "_debug"
    if write_debug_visualizations:
        debug_dir.mkdir(parents=True, exist_ok=True)

    requested_static_categories = requested_static_categories or set()
    dynamic_boxes_per_frame = dynamic_boxes_per_frame or {}

    results: List[Dict[str, Any]] = []

    for frame_path in frame_list:
        frame_path = str(frame_path)
        frame_name = Path(frame_path).name
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)

        if img is None:
            results.append({
                "frame": frame_name,
                "mask_path": "",
                "usable_area_pct": 0.0,
                "masked_categories": [],
                "error": "unreadable_frame",
            })
            continue

        try:
            dynamic_boxes = dynamic_boxes_per_frame.get(frame_name, [])
            keep_mask, metrics, debug_img = build_mask_for_frame(
                img, requested_static_categories, dynamic_boxes
            )

            mask_path = masks_dir / f"{Path(frame_path).stem}__mask.png"
            cv2.imwrite(str(mask_path), keep_mask)

            if write_debug_visualizations:
                debug_path = debug_dir / f"{Path(frame_path).stem}__debug.jpg"
                cv2.imwrite(str(debug_path), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        except Exception:
            logger.error("Fallo generando máscara para %s; se omite.", frame_path, exc_info=True)
            results.append({
                "frame": frame_name,
                "mask_path": "",
                "usable_area_pct": 0.0,
                "masked_categories": [],
                "error": "mask_generation_failed",
            })
            continue

        row = {"frame": frame_name, "mask_path": str(mask_path), **metrics}
        results.append(row)

    return results
