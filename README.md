
# Automated Quality Inspection of Frozen-Meat Packaging

## MMA3001: Numerical Methods and Machine Learning — Final Project

---

## Engineering Problem & Operational Context

### Problem Context

In automated high-speed food processing and packaging lines, maintaining seal integrity and container quality is essential. For perishable meat products (such as  pork rashers), packaging defects can lead to many negative consequences such as monetary losses and health concerns.

### Intended Operational Use

This repository provides an automated, reproducible computer vision workflow designed for real-time quality assurance on meat packaging conveyor lines:

1. **Primary Model (YOLOv11 Object Detection)**: Detects, localizes, and classifies multiple simultaneous packaging defects (*Loose Meat*, *Packaging Error*, *Twisted Meat*, *Unsealed*, *Wrinkle*) with spatial bounding boxes and confidence scores.
2. **Alternative (Scikit-Learn MLP)**: Performs rapid binary defect classification (*Damaged* vs. *Undamaged*) to evaluate whether full spatial localization earns its computational complexity compared to a lightweight whole-image classifier.

---

## Features

- **Multi-Class Defect Detection**: Detects and localizes 5 defect categories (`Loose Meat`, `Packaging Error`, `Twisted Meat`, `Unsealed`, `Wrinkle`) alongside undamaged packages.
- **Dual Pipeline Support**: Run full YOLOv11 object detection or lightweight binary classification.
- **Batch & Single-Image Inference**: Command-line tools to evaluate individual images or process entire directories, exporting results to structured JSON and annotated images.
- **Automated Verification**: Full unit test coverage across data parsing, model training interfaces, and inference pipelines.

---

## Installation

### Prerequisites
- Python 3.9+
- Recommended: Dedicated virtual environment

```bash
git clone https://github.com/<your-username>/packaging-defect-detection.git
cd packaging-defect-detection

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Dataset Inspection & Visualisation

Inspect class distribution across splits:
```bash
python3 dataset_info.py
```

Visualize sample annotated images with bounding boxes:
```bash
python3 visualisation.py --n 6 --save "packaging_defect_samples.png"
```

---

### 2. Primary Pipeline: YOLOv11

#### Train
```bash
python3 training.py --epochs 50 --batch 16 --imgsz 640
```

#### Evaluate on Test Set
```bash
# Standard bounding-box evaluation (mAP50, mAP50-95)
python3 testing.py --split test

# Custom image-level accuracy & confusion matrix breakdown
python3 evaluate_yolo_model.py
```

#### Run Inference
Run predictions on a single image:
```bash
python3 inference.py "path/to/image.jpg" --weights "runs/detect/pork_rasher_detection/weights/best.pt"
```

Run batch inference across the entire test set and save predictions:
```bash
python3 inference.py "Pork Rasher Error-Packaging-.v4i.yolov11/test/images" \
  --weights "runs/detect/pork_rasher_detection/weights/best.pt" \
  --output-json "runs/final/final.json" \
  --save-dir "runs/final/images"
```

---

### 3. Baseline Pipeline: Scikit-Learn MLP

#### Train & Evaluate
```bash
python3 train_packaging_nn-v2.py \
  --dataset_dir "Pork Rasher Error-Packaging-.v4i.yolov11" \
  --output_dir "Binary_Classification/results" \
  --model_name "binary_model"
```
Outputs model weights (`binary_model.joblib`), evaluation summaries, and confusion matrix plots to `Binary_Classification/results/`.

#### Run Inference
```bash
# Single image (resolution is auto-inferred from model)
python3 predict_damage-v2.py "path/to/image.jpg"

# Batch folder
python3 predict_damage-v2.py \
  --dir "Pork Rasher Error-Packaging-.v4i.yolov11/test/images" \
  --output_json "Binary_Classification/results/predictions.json"
```

---

## Testing

Run the automated test suite with `pytest`:
```bash
pytest -v
```
Or using Python's standard `unittest` test runner:
```bash
python3 -m unittest discover -s . -p "test_*.py"
```

---

## Directory Hierarchy

The dataset and project files are organised as follows:

```text
.
├── Pork Rasher Error-Packaging-.v4i.yolov11/
│   ├── data.yaml                  # Paths, split definitions, and class names
│   ├── README.dataset.txt         # Overview of the dataset
│   ├── README.roboflow.txt        # Overview of the source of data
│   ├── train/
│   │   ├── images/                # Training RGB images (.jpg, .jpeg, .png)
│   │   └── labels/                # YOLO normalized bounding box .txt files
│   ├── valid/
│   │   ├── images/                # Validation split for hyperparameter tuning
│   │   └── labels/
│   └── test/
│       ├── images/                # Withheld evaluation split for unbiased benchmarking
│       └── labels/
│
├── Tests/                         # Automated unit test suite (pytest / unittest)
│   ├── test_dataset_info.py
│   ├── test_visualisation.py
│   ├── test_evaluate_yolo_model.py
│   ├── test_training_testing_yolo.py
│   └── test_reports/              # Automated results from test suite
│
├── runs/                          # YOLO Model Outputs
│   ├── detect/
│   │   ├── pork_rasher_detection/ # Training outputs
│   │   └── val/                   # Validation outputs
│   └── Final_Test/                # Inference outputs on test data
│
├── Binary_Classification/         # Scikit-Learn MLP
│   ├── dataset.py                 # Label parsing, feature extraction, and split loaders
│   ├── model-v2.py                # Scikit-Learn MLP pipeline, training, and evaluation logic
│   ├── train_packaging_nn-v2.py   # Main training script; outputs model and evaluation reports
│   ├── predict_damage-v2.py       # Standalone inference engine (auto-infers resolution)
│   ├── test_packaging_nn-v2.py    # Tests for functions used for MLP
│   ├── test_predict_damage-v2.py  # Tests for functions used for MLP
│   ├── results/                    # Contains MLP model and validation and testing results
│   └── test_reports/              # Automated results from test suite
│
├── dataset_info.py                # Class distribution and defect instance tallying
├── visualisation.py               # Dataset visualization with color-coded bounding boxes
├── training.py                    # Ultralytics YOLOv11 training script
├── testing.py                     # Standard YOLO test split evaluation (mAP50)
├── evaluate_yolo_model.py         # Custom image-level binary & 6-class confusion matrix evaluation
├── inference.py                   # Inferences images of folder of files and runs it through model
├── yolo11n.pt                     # Base Ultralytics YOLOv11 nano model
│
├── docs/                          # Auto-generated HTML documentation (via pdoc)
│
├── requirements.txt               # Package dependencies
├── LICENSE                        # MIT Open Source License
└── README.md                      # Project overview and workflow documentation
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.