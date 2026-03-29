# Body Shape Biometric Recognition

A soft biometric identification system that recognises individuals from full-length photographs using body shape and silhouette analysis. Combines YOLOv8 detection, SAM2 segmentation, and KeypointRCNN pose estimation to extract rich geometric and statistical features for KNN/SVM classification.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellow)

---

## Pipeline

```
Input Image
    |
    v
YOLO v8 (person detection)
    |
    v
SAM 2.1 (instance segmentation -> binary silhouette)
    |
    v
KeypointRCNN (18 COCO keypoints -> OpenPose format)
    |
    v
Feature Extraction (8 feature types, view-dependent)
    |-- measurements    (pixel distances between landmarks)
    |-- silhouette_height
    |-- hu_silhouette   (Hu moments of body contour)
    |-- angles          (joint angles)
    |-- hu_head         (Hu moments of head region)
    |-- pca             (PCA on silhouette)
    |-- hog             (Histogram of Oriented Gradients)
    |-- raycast         (horizontal profile of silhouette)
    |
    v
KNN / SVM Classification (per-view: frontal + side)
    |
    v
Evaluation: CCR + EER curves
```

---

## Results

Evaluated on a 21-subject dataset with frontal and side views.

| View | CCR (Recognition) | EER (Verification) |
|------|------------------|--------------------|
| Frontal | 64% | ~30% |
| Side | 64% | ~28% |
| Combined (weighted) | 64% | ~29% |

---

## Setup

### Prerequisites

- Python 3.10+
- CMake and a C++ compiler (required by dlib)
  - macOS: `brew install cmake`
  - Ubuntu: `sudo apt install cmake build-essential`

### Install

```bash
git clone https://github.com/Sik11/body-shape-biometric-recognition.git
cd body-shape-biometric-recognition
pip install -r requirements.txt
```

PyTorch with GPU support can be installed separately from [pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`.

### Download dlib models

The head-region extractor uses two dlib models that must be downloaded manually:

```bash
# CNN face detector (~67 MB)
wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
bzip2 -d mmod_human_face_detector.dat.bz2

# 68-point shape predictor (~99 MB)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

YOLO and SAM weights are downloaded automatically by ultralytics on first run.

---

## Usage

```bash
# Basic run (uses default paths: datasets/training, datasets/test)
python run.py

# Specify dataset paths explicitly
python run.py --train path/to/training --test path/to/test

# Use a cached feature extractor to skip reprocessing
python run.py --cache my_cache.pickle

# Custom dlib model paths
python run.py --dlib-detector /path/to/mmod_human_face_detector.dat \
              --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat

# All options
python run.py --help
```

### Dataset format

Training images must follow the naming convention `{SUBJECT_ID}f.jpg` (frontal) and `{SUBJECT_ID}s.jpg` (side). See `datasets/README.md` for full details and directory layout.

### Outputs

All generated files are written to `outputs/`:
- `outputs/predictions.txt` - per-image classification results
- `outputs/eer_curves/` - EER plots for frontal, side, and combined views
- `outputs/histograms/` - intra/inter-class distance histograms

---

## Repository structure

```
.
├── run.py                  # Entry point with argparse CLI
├── src/
│   ├── analyser.py         # HumanBodyAnalyser: segmentation, pose, feature extraction
│   ├── classifier.py       # KNNClassifiers, SVMClassifiers, UnifiedSVMClassifier
│   └── utils.py            # Shared visualization helpers
├── datasets/
│   └── README.md           # Instructions for obtaining dataset
├── outputs/                # Generated at runtime (gitignored)
└── requirements.txt
```

---

## License

MIT. See [LICENSE](LICENSE).

Dataset images are not included in this repository. Users must obtain their own copy of the Southampton Gait Database.
