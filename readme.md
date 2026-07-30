# intelliframes4colmap
## Fotogramas Inteligentes para COLMAP

**Optimización de flujos de trabajo pre-COLMAP para la creación de Gaussian Splats de alta fidelidad.**

---

# 1. Visión del proyecto

**intelliframes4colmap** es una herramienta de análisis inteligente diseñada para actuar como un filtro de calidad y optimización previo al proceso de fotogrametría.

Su objetivo principal es evitar horas de procesamiento desperdiciadas durante la generación de **Gaussian Splats**, seleccionando únicamente los fotogramas óptimos y generando automáticamente la configuración más adecuada para **COLMAP**.

La herramienta combina:

- Visión por computador clásica
- Deep Learning
- Modelos de Lenguaje Visual (VLM)

para:

- analizar la semántica de la escena;
- detectar imágenes desenfocadas;
- eliminar fotogramas redundantes;
- identificar movimiento de objetos;
- estimar la calidad de la captura;
- recomendar automáticamente los parámetros óptimos para COLMAP.

El objetivo es conseguir reconstrucciones 3D más robustas desde el primer procesamiento.

---

# 2. Salidas del sistema

El analizador genera una estructura completa de datos para automatizar pipelines de reconstrucción.

```
project/
│
├── report.html
├── analysis.json
│
├── frames_selected/
├── frames_rejected/
├── thumbnails/
│
├── motion.csv
├── sharpness.csv
├── exposure.csv
├── texture.csv
│
├── masks/
│
└── colmap_config.json
```

## Reporte visual

**report.html**

Resumen visual del análisis para el usuario.

---

## Datos de análisis

**analysis.json**

Salida estructurada para integración con scripts Python, Bash u otros pipelines.

---

## Gestión de fotogramas

### frames_selected/

Fotogramas seleccionados por presentar:

- máxima nitidez
- mayor riqueza de textura
- mejor cobertura espacial

### frames_rejected/

Fotogramas descartados por:

- desenfoque
- motion blur
- baja textura
- redundancia excesiva

### thumbnails/

Miniaturas para validación rápida.

---

## Métricas

- motion.csv
- sharpness.csv
- exposure.csv
- texture.csv

---

## Máscaras

```
masks/
```

Segmentación de objetos dinámicos o regiones no relevantes.

---

## Configuración COLMAP

```
colmap_config.json
```

o

```
colmap_config.yaml
```

Contiene todos los parámetros recomendados para el proceso de matching y reconstrucción.

---

# 3. Etapas del análisis inteligente

# Nivel 1 — Calidad de imagen

---

## 1.1 Sharpness Score

Cada fotograma recibe un índice de nitidez utilizado para eliminar:

- motion blur
- desenfoque
- imágenes excesivamente comprimidas

Estas imágenes generan artefactos durante la reconstrucción del Gaussian Splat.

### Método

1. Se calcula el Laplaciano.
2. Se obtiene su imagen integral.
3. Se aplica una ventana deslizante (`sharpnessWindowSize`).
4. Se calcula la desviación estándar local.
5. El valor máximo encontrado se utiliza como **Sharpness Score**.

Si existen máscaras, las regiones enmascaradas no participan en el cálculo.

---

## 1.2 Estimación dinámica del solapamiento

En lugar de utilizar un valor fijo:

```
overlap = 10
```

el sistema estima automáticamente la similitud entre fotogramas mediante la persistencia de características.

Ejemplo:

| Comparación | Features comunes |
|-------------|-----------------|
| Frame A ↔ Frame B | 74 % |
| Frame A ↔ Frame C | 41 % |

A partir de esta información se calcula una curva temporal de redundancia.

Ejemplos de recomendación:

Movimiento lento

```
Sequential overlap = 14
```

Movimiento rápido

```
Sequential overlap = 25
```

---

## 1.3 Optical Flow

Se analiza el desplazamiento de píxeles entre fotogramas consecutivos.

### Movimiento de cámara

```
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

Todo el escenario se mueve de forma coherente.

---

### Movimiento de objetos

```
>>>>>>>>>   PERSONA   <<<<<<<<

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

Solo determinados objetos presentan movimiento.

Estos objetos serán enmascarados para evitar errores durante el matching.

### Recomendación automática

Trayectoria continua

```
Sequential Matcher
```

Trayectoria con bucles

```
Loop Detection = ON
Vocabulary Tree = ON
```

---

# Nivel 2 — Análisis semántico

---

## 2.1 Segmentación

Mediante modelos de segmentación se detectan:

- sky
- glass
- water
- vegetation
- animals
- birds
- people
- cars
- ground
- uniform walls

Estas regiones pueden excluirse del proceso de reconstrucción.

Beneficios:

- menos keypoints erróneos
- mayor velocidad
- mejor calidad final

---

## 2.2 Texture Richness

Se calcula la densidad de características por megapíxel.

Ejemplo:

### Alta textura

```
Pared de ladrillo

Texture = HIGH

SIFT max features = 16000
```

### Baja textura

```
Pared lisa

Texture = LOW

SIFT max features = 8000
```

---

## 2.3 Control de exposición

Se analizan:

- luminancia
- histogramas
- clipping
- altas luces
- sombras

Cuando la variación de exposición es elevada:

```
Tone Mapping = ON
```

para evitar cambios visibles de brillo en el Gaussian Splat.

---

## 2.4 Caracterización óptica

Conocer correctamente:

- distancia focal
- tipo de lente
- cámara utilizada

reduce enormemente el espacio de búsqueda de parámetros intrínsecos de COLMAP.

Ejemplo:

```
24 mm
```

vs

```
35 mm
```

Esto evita muchos errores de triangulación que aparecen tras varias horas de procesamiento.

---

## 2.5 Sincronización GNSS / IMU

El sistema puede utilizar telemetría procedente de:

- DJI
- Smartphones
- GPS industriales

### Extracción

Datos obtenidos desde:

- GPX
- CSV
- LOG
- Metadatos del vídeo

### Sincronización

El GPS suele trabajar entre:

```
1–10 Hz
```

mientras el vídeo puede grabarse a:

```
30–60 FPS
```

Se utiliza interpolación mediante **Spline Cúbica** para asignar a cada fotograma:

- X
- Y
- Z
- Roll
- Pitch
- Yaw

Finalmente se convierten las coordenadas geodésicas a coordenadas cartesianas locales.

---

# Nivel 3 — Decision Engine

El resultado final es un archivo:

```
colmap_config.json
```

que contiene la estrategia óptima para COLMAP.

Entre otros parámetros:

- Matching Strategy
- Sequential / Quadratic
- Loop Detection
- Vocabulary Tree
- SIFT Max Features
- Camera Model
- Guided Matching
- Tone Mapping

---

# 4. Ejemplo de reporte

```text
intelliframes4colmap | VIDEO ANALYSIS REPORT
===========================================

[METADATA]

Resolution ............ 3840 × 2160
FPS ................... 59.94
Duration .............. 02:18
Frames ................ 8276

[SCENE ANALYSIS]

Camera movement ....... HIGH
Rotation .............. MEDIUM/HIGH
Translation ........... HIGH
Motion Blur ........... MEDIUM
Exposure variation .... LOW
Texture density ....... HIGH
Repeated texture ...... LOW
Dynamic objects ....... LOW

[FRAME SELECTION]

Recommended ........... 1247
Rejected .............. 7029

[COLMAP]

Matching .............. Sequential
Overlap ............... 18
Loop Detection ........ ON
Vocabulary Tree ....... Recommended
SIFT Features ......... 12000
Camera Model .......... SIMPLE_RADIAL
Guided Matching ....... ON
Mapper ................ Incremental
Tone Mapping .......... OFF

[WARNINGS]

00:31:25 → 00:39:10

Camera rotation is very slow.
Frame redundancy HIGH.
Selected 14 / 480 frames.

01:04:01 → 01:07:20

Camera movement increases sharply.
Selected 31 / 180 frames.

01:21:22 → 01:23:00

Motion blur detected.
63% of frames exceed the quality threshold.
```

---

# 5. Toolbox

## Core Engine

- FFmpeg
- FFprobe
- OpenCV
- AliceVision
- Segment Anything Model (SAM)
- YOLO

---

## Datos y Geoespacial

- ExifTool
- SciPy
- PyProj
- Pandas

---

## Reconstrucción

- COLMAP 4.1

---

# Objetivo final

**intelliframes4colmap** pretende convertirse en un orquestador inteligente capaz de analizar automáticamente cualquier vídeo antes de su procesamiento fotogramétrico, generando una selección óptima de fotogramas y una configuración personalizada para COLMAP.

El objetivo es reducir drásticamente los tiempos de prueba y error, mejorar la calidad de los Gaussian Splats y minimizar reconstrucciones fallidas tras horas de procesamiento.
