# intelliframes4colmap
## Smart Frames for COLMAP

**Optimization of pre-COLMAP workflows for creating high-fidelity Gaussian Splats.**

---

# 1. Project Vision

**intelliframes4colmap** is an intelligent analysis tool designed to act as a quality filter and optimization step prior to the photogrammetry process.

Its main goal is to prevent wasted hours of processing during **Gaussian Splats** generation by selecting only the optimal frames and automatically generating the most suitable configuration for **COLMAP**.

The tool combines:

- Classical Computer Vision
- Deep Learning
- Visual Language Models (VLM)

to:

- analyze the semantics of the scene;
- detect blurry images;
- remove redundant frames;
- identify object motion;
- estimate capture quality;
- automatically recommend optimal parameters for COLMAP.

The aim is to achieve more robust 3D reconstructions from the first processing attempt.

---

# 2. System Outputs

The analyzer generates a complete data structure to automate reconstruction pipelines.

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

## Visual Report

**report.html**

Visual summary of the analysis for the user.

---

## Analysis Data

**analysis.json**

Structured output for integration with Python, Bash, or other pipeline scripts.

---

## Frame Management

### frames_selected/

Frames selected for having:

- maximum sharpness
- highest texture richness
- best spatial coverage

### frames_rejected/

Frames discarded due to:

- blur
- motion blur
- low texture
- excessive redundancy

### thumbnails/

Thumbnails for quick validation.

---

# 3. COLMAP Configuration

The tool automatically generates a **colmap_config.json** file with optimized parameters:

- Matching Strategy
- Overlap
- Loop Detection
- Vocabulary Tree
- SIFT Max Features
- Camera Model
- Guided Matching
- Tone Mapping

---

# 4. Example Report

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

## Data and Geospatial

- ExifTool
- SciPy
- PyProj
- Pandas

---

## Reconstruction

- COLMAP 4.1

---

# Final Goal

**intelliframes4colmap** aims to become an intelligent orchestrator capable of automatically analyzing any video before its photogrammetric processing, generating an optimal selection of frames and a customized configuration for COLMAP.

The goal is to drastically reduce trial-and-error times, improve the quality of Gaussian Splats, and minimize failed reconstructions after hours of processing.
