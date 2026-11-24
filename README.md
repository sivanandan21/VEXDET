# WorkZone Vision

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/sivanandan21/workzone-vision/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Edge-ready CCTV person-detection fine-tuning, hard-negative mining, and ONNX export pipeline built on Apache-2.0 RF-DETR architectures.

Author: **sivanandan21** ([sivapn721@gmail.com](mailto:sivapn721@gmail.com))

---

## Overview

**WorkZone Vision** is a specialized engineering toolkit for optimizing person and head detection accuracy in office CCTV and workspace monitoring environments. It provides an end-to-end workflow to:

1. **Download & Initialize**: Fetch Apache-2.0 RF-DETR architectures (`RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`).
2. **Quality Frame Extraction**: Filter out motion blur (Laplacian variance) and static duplicate scenes (frame difference & histogram correlation) from raw CCTV footage.
3. **Dataset Validation**: Validate COCO annotation integrity, bounding box geometry, image readability, and class distributions.
4. **Fine-Tuning**: Adapt the detector to custom office CCTV data (`person` or dual-class `person` + `head`) on CUDA or CPU backends.
5. **Hard-Negative Mining**: Identify ambiguous detections (confidence 0.15–0.45) and false positives for targeted retraining cycles.
6. **Rigorous Benchmarking**: Compute COCO-standard `mAP@50`, `mAP@50:95`, precision, recall, and inference latency (`ms`, FPS) comparing baseline against fine-tuned checkpoints.
7. **ONNX Export & Verification**: Export fine-tuned models to ONNX and verify cross-backend runtime accuracy.

---

## Project Structure

```
workzone-vision/
├── configs/
│   └── config.yaml               # Hyperparameters and pipeline thresholds
├── data/
│   ├── raw_videos/.gitkeep       # CCTV video clips (.mp4, .avi, etc.)
│   ├── extracted_frames/.gitkeep # Filtered sharp candidate frames
│   ├── dataset/.gitkeep          # Labeled COCO dataset (train, val, test)
│   └── hard_negatives/.gitkeep   # Mined ambiguous/false-positive frames
├── models/
│   ├── checkpoints/.gitkeep      # PyTorch fine-tuning checkpoints
│   └── exported/.gitkeep         # Exported ONNX models
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic v2 schemas and model validation
│   ├── model_factory.py          # Model and inference predictor factory
│   ├── download_model.py         # Model download & verification routine
│   ├── extract_frames.py         # Laplacian blur & duplicate scene filter
│   ├── validate_dataset.py       # COCO structure and box geometry validator
│   ├── train_finetune.py         # Model fine-tuning execution
│   ├── mine_hard_negatives.py    # Hard-negative & ambiguous detection miner
│   ├── evaluate.py               # Precision-recall curves, mAP & latency benchmark
│   └── export_model.py           # ONNX export and runtime verification
├── tests/
│   └── test_smoke.py             # Comprehensive test suite
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI matrix
├── cli.py                        # Unified command-line interface
├── pyproject.toml                # Project packaging and entry points
├── requirements.txt              # Pinned requirements
├── LICENSE                       # Apache-2.0 License
└── README.md                     # Documentation
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA drivers (optional; automatically falls back to CPU if unavailable)

### Install via pip
```bash
git clone https://github.com/sivanandan21/workzone-vision.git
cd workzone-vision
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

---

## Usage

WorkZone Vision can be executed via the installed `workzone` command or `python cli.py`:

```bash
workzone --help
```

### 1. Download Model
Fetch official weights for `RFDETRSmall` (or `nano`, `medium`, `large`):
```bash
workzone download --model-name RFDETRSmall
```

### 2. Extract & Filter CCTV Frames
Extract clear candidate frames from video clips in `data/raw_videos/`:
```bash
workzone extract-frames --video-dir data/raw_videos --frame-interval 15 --blur-threshold 100.0
```
- Drops blurry frames (`blur_score < blur_threshold`).
- Drops static scenes (`change_pct < 1.5%` or histogram correlation `>= 0.95`).

### 3. Validate Dataset
Verify your labeled COCO dataset before training:
```bash
workzone validate-dataset --dataset-dir data/dataset --classes person
```

### 4. Fine-Tune Model
Train on your custom office CCTV dataset:
```bash
workzone train --model-name RFDETRSmall --epochs 15 --batch-size 4 --lr 0.0001
```

### 5. Mine Hard Negatives
Extract ambiguous frames (confidence 0.15–0.45) for human review and retraining round 2:
```bash
workzone mine-negatives --input-dir data/extracted_frames --min-conf 0.15 --max-conf 0.45
```
- Saves raw frames to `data/hard_negatives/images/`.
- Saves annotated visuals to `data/hard_negatives/visualizations/`.
- Generates `mining_manifest.json` for review.

### 6. Benchmark Accuracy & Latency
Evaluate the fine-tuned checkpoint against baseline pretrained weights:
```bash
workzone evaluate --model-name RFDETRSmall --checkpoint models/checkpoints/best.pt
```
Computes:
- `mAP@50` (Mean Average Precision at IoU 0.50)
- `mAP@50:95` (COCO standard mean Average Precision across IoU thresholds 0.50:0.95 in 0.05 steps)
- Latency (min, max, mean, P95 in `ms`) and FPS throughput
- Side-by-side comparison with absolute improvement deltas

### 7. Export to ONNX
Export the model to ONNX format and run verification inference:
```bash
workzone export --checkpoint models/checkpoints/best.pt --model-name RFDETRSmall
```

### Full End-to-End Pipeline
Run all stages sequentially:
```bash
workzone pipeline --model-name RFDETRSmall
```

---

## Testing

Run the test suite using pytest:
```bash
pytest -v
```

Tests cover:
- Architecture whitelist validation
- Laplacian blur score and duplicate frame differentiation
- Synthetic COCO dataset validation (valid, missing images, invalid coordinates)
- Bounding box IoU calculations and 101-point Average Precision interpolation
- BGR to RGB color conversion for inference

---

## Third-Party Attributions

This project relies on and acknowledges the following open-source libraries:
- **[RF-DETR](https://github.com/roboflow/rf-detr)** (Apache-2.0 License): Real-time detection transformer model architectures.
- **[Supervision](https://github.com/roboflow/supervision)** (Apache-2.0 License): Computer vision utilities and detection data structures.
- **[OpenCV](https://opencv.org/)** (Apache-2.0 License): Image and video processing.
- **[ONNX Runtime](https://onnxruntime.ai/)** (MIT License): Cross-platform, high-performance ML inference engine.
- **[PyTorch](https://pytorch.org/)** (Modified BSD License): Deep learning framework.
- **[Typer](https://typer.tiangolo.com/)** & **[Rich](https://github.com/Textualize/rich)** (MIT License): Command-line interface and terminal formatting.

---

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
