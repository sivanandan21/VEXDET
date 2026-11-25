"""Smoke test suite for VEXDET pipeline."""

import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pytest

from src.config import ALLOWED_MODELS, ModelConfig, PipelineConfig, load_config
from src.evaluate import (
    coco_xywh_to_xyxy,
    compute_average_precision,
    compute_box_iou,
    evaluate_dataset_accuracy,
)
from src.extract_frames import compute_blur_score, compute_frame_difference
from src.model_factory import (
    ALLOWED_MODEL_NAMES,
    create_class_thresholds,
    get_rfdetr_class,
)
from src.predict import draw_bounding_boxes, run_predict
from src.validate_dataset import validate_split


# ==========================================
# 1. MODEL WHITELIST VALIDATION
# ==========================================

def test_allowed_models_whitelist():
    """Verify that only approved Apache-2.0 RF-DETR models are permitted."""
    expected_allowed = {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"}
    assert ALLOWED_MODELS == expected_allowed
    assert set(ALLOWED_MODEL_NAMES) == expected_allowed


@pytest.mark.parametrize("valid_model", ["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"])
def test_valid_model_config(valid_model: str):
    """Verify all approved Apache-2.0 models pass config validation."""
    cfg = ModelConfig(name=valid_model)
    assert cfg.name == valid_model


@pytest.mark.parametrize("invalid_model", [
    "RFDETRAtto",
    "RFDETRFemto",
    "RFDETRPico",
    "RFDETRXLarge",
    "RFDETR2XLarge",
    "YOLOv8",
    "ultralytics",
    "YOLOv10",
    "YOLO11",
])
def test_invalid_and_non_apache_models_rejected(invalid_model: str):
    """Verify non-Apache models and AGPL/Ultralytics models are rejected."""
    with pytest.raises(ValueError) as excinfo:
        ModelConfig(name=invalid_model)
    assert "Allowed Apache-2.0 architectures" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo_factory:
        get_rfdetr_class(invalid_model)
    assert "Allowed Apache-2.0 models" in str(excinfo_factory.value)


# ==========================================
# 2. CONFIGURATION TESTS
# ==========================================

def test_pipeline_config_defaults(tmp_path: Path):
    """Verify default config initialization."""
    cfg = PipelineConfig()
    assert cfg.model.name == "RFDETRLarge"
    assert cfg.preprocess.enable_clahe is True
    assert cfg.training.epochs == 30
    assert cfg.training.batch_size == 2
    assert cfg.training.grad_accum_steps == 8
    assert cfg.training.lr == 0.00008
    assert "person" in cfg.detection.classes
    assert cfg.extraction.blur_threshold == 100.0
    assert cfg.hard_mining.min_confidence == 0.15
    assert cfg.hard_mining.max_confidence == 0.45


def test_invalid_confidence_range():
    """Verify that min_confidence >= max_confidence raises validation error."""
    with pytest.raises(ValueError):
        PipelineConfig(
            hard_mining={
                "min_confidence": 0.50,
                "max_confidence": 0.30,
            }
        )


def test_class_thresholds_creation():
    """Verify ClassThresholds creation helper."""
    th = create_class_thresholds({"person": 0.35, "head": 0.40}, default_threshold=0.50)
    assert th is not None
    if hasattr(th, "get"):
        assert th.get("person") == 0.35
        assert th.get("head") == 0.40
        assert th.get("car", 0.50) == 0.50


# ==========================================
# 3. FRAME EXTRACTION QUALITY FILTERS TESTS
# ==========================================

def test_blur_metric_sharp_vs_blurred():
    """Verify Laplacian variance correctly differentiates sharp from blurry frames."""
    # Create sharp synthetic image with sharp geometric edges
    sharp_img = np.zeros((400, 400), dtype=np.uint8)
    for i in range(0, 400, 20):
        sharp_img[i:i + 10, :] = 255
        sharp_img[:, i:i + 10] = 255

    # Create blurred image with strong Gaussian kernel
    blurred_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)

    sharp_score = compute_blur_score(sharp_img)
    blurred_score = compute_blur_score(blurred_img)

    assert sharp_score > 1000.0, f"Expected high score for sharp image, got {sharp_score}"
    assert blurred_score < 100.0, f"Expected low score for blurred image, got {blurred_score}"
    assert sharp_score > blurred_score * 10


def test_duplicate_frame_detection():
    """Verify identical frames produce ~0% change and high correlation."""
    img1 = np.full((300, 300), 128, dtype=np.uint8)
    img2 = img1.copy()
    img3 = img1.copy()
    # Add significant motion in img3
    img3[50:250, 50:250] = 255

    # Identical frames
    change_pct_ident, corr_ident = compute_frame_difference(img1, img2)
    assert change_pct_ident == 0.0
    assert corr_ident >= 0.999

    # Distinct frames
    change_pct_diff, _ = compute_frame_difference(img1, img3)
    assert change_pct_diff > 40.0


# ==========================================
# 4. DATASET VALIDATOR TESTS
# ==========================================

def test_validate_split_synthetic_valid(tmp_path: Path):
    """Test dataset validator on a valid synthetic COCO split."""
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    # Create 2 valid dummy images
    img1_path = split_dir / "frame_001.jpg"
    img2_path = split_dir / "frame_002.jpg"
    cv2.imwrite(str(img1_path), np.zeros((640, 640, 3), dtype=np.uint8))
    cv2.imwrite(str(img2_path), np.zeros((640, 640, 3), dtype=np.uint8))

    # Create valid COCO JSON
    coco_data = {
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 640},
            {"id": 2, "file_name": "frame_002.jpg", "width": 640, "height": 640},
        ],
        "categories": [
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 150], "area": 7500, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [200, 200, 60, 120], "area": 7200, "iscrowd": 0},
        ],
    }

    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    res = validate_split(
        split_name="train",
        split_dir=split_dir,
        annotation_path=ann_path,
        target_classes=["person"],
    )

    assert res["valid"] is True
    assert len(res["errors"]) == 0
    assert res["stats"]["valid_boxes"] == 2
    assert res["stats"]["class_counts"]["person"] == 2


def test_validate_split_missing_image(tmp_path: Path):
    """Test dataset validator catches missing image files on disk."""
    split_dir = tmp_path / "val"
    split_dir.mkdir()

    coco_data = {
        "images": [
            {"id": 1, "file_name": "non_existent.jpg", "width": 640, "height": 640},
        ],
        "categories": [
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 50, 50], "area": 2500, "iscrowd": 0},
        ],
    }

    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    res = validate_split(
        split_name="val",
        split_dir=split_dir,
        annotation_path=ann_path,
        target_classes=["person"],
    )

    assert res["valid"] is False
    assert any("not found on disk" in err for err in res["errors"])


# ==========================================
# 5. EVALUATION MATHEMATICS TESTS
# ==========================================

def test_compute_box_iou():
    """Verify IoU calculation on known coordinate geometries."""
    # Identical boxes
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [0.0, 0.0, 10.0, 10.0]
    assert compute_box_iou(box_a, box_b) == 1.0

    # Non-overlapping boxes
    box_c = [20.0, 20.0, 30.0, 30.0]
    assert compute_box_iou(box_a, box_c) == 0.0

    # 50% overlap geometry:
    # box1: [0, 0, 10, 10] -> area 100
    # box2: [5, 0, 15, 10] -> area 100, inter = [5, 0, 10, 10] -> area 50
    # union = 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    box_d = [5.0, 0.0, 15.0, 10.0]
    assert abs(compute_box_iou(box_a, box_d) - (1.0 / 3.0)) < 1e-5


def test_coco_xywh_to_xyxy():
    """Verify box coordinate conversion from [x, y, w, h] to [x1, y1, x2, y2]."""
    xywh = [10.0, 20.0, 30.0, 40.0]
    xyxy = coco_xywh_to_xyxy(xywh)
    assert xyxy == [10.0, 20.0, 40.0, 60.0]


def test_average_precision_calculation():
    """Verify 101-point interpolated Average Precision on ideal detector output."""
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ap = compute_average_precision(recalls, precisions)
    assert abs(ap - 1.0) < 1e-4

    # Empty detector
    ap_empty = compute_average_precision(np.array([]), np.array([]))
    assert ap_empty == 0.0


# ==========================================
# 6. COLOR CONVERSION REQUIREMENT TEST
# ==========================================

def test_opencv_color_conversion_rule():
    """Verify OpenCV BGR to RGB color conversion operates correctly."""
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Set blue channel to 255
    bgr_frame[:, :, 0] = 255

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    # In RGB, red is at channel 0, blue at channel 2
    assert rgb_frame[:, :, 0].max() == 0
    assert rgb_frame[:, :, 2].max() == 255


# ==========================================
# 7. HIGH-ACCURACY MODULES TESTS
# ==========================================

from src.preprocess import enhance_cctv_frame
from src.enhanced_inference import (
    compute_iou_matrix,
    fuse_boxes_wbf,
    generate_image_slices,
    VexdetEnhancedPredictor,
    WorkZoneEnhancedPredictor,
)


def test_clahe_enhancement():
    """Verify CLAHE enhancement preserves shape and dynamic range."""
    dark_img = np.full((120, 160, 3), 30, dtype=np.uint8)
    dark_img[40:80, 40:80, :] = 45
    enhanced = enhance_cctv_frame(dark_img, clip_limit=2.0)
    assert enhanced.shape == dark_img.shape
    assert enhanced.dtype == np.uint8
    assert enhanced.mean() >= dark_img.mean()


def test_generate_image_slices():
    """Verify image slicing generates overlapping windows covering canvas."""
    slices = generate_image_slices(1080, 1920, slice_size=640, overlap_ratio=0.25)
    assert len(slices) > 1
    for x1, y1, x2, y2 in slices:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080


def test_fuse_boxes_wbf():
    """Verify Weighted Boxes Fusion merges overlapping detections into centroid."""
    boxes = [
        [100.0, 100.0, 200.0, 300.0],
        [104.0, 98.0, 202.0, 305.0],
        [500.0, 500.0, 600.0, 700.0],
    ]
    scores = [0.80, 0.90, 0.75]
    class_ids = [0, 0, 0]
    class_names = ["person", "person", "person"]

    fused = fuse_boxes_wbf(boxes, scores, class_ids, class_names, iou_threshold=0.55)
    assert len(fused["boxes"]) == 2
    assert fused["scores"][0] > 0.85


def test_workzone_enhanced_predictor():
    """Verify VexdetEnhancedPredictor (and WorkZone alias) wraps base predictor with CLAHE, TTA, Slicing & WBF."""
    class MockPredictor:
        def predict(self, img, threshold=0.35):
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            box = np.array([[cx - 20, cy - 40, cx + 20, cy + 40]], dtype=np.float32)
            conf = np.array([0.70], dtype=np.float32)
            cids = np.array([0], dtype=np.int32)
            class _MockDets:
                xyxy = box
                confidence = conf
                class_id = cids
                data = {"class_name": ["person"]}
            return _MockDets()

    mock = MockPredictor()
    enhanced_p = VexdetEnhancedPredictor(
        base_predictor=mock,
        enable_clahe=True,
        enable_tiling=True,
        enable_tta=True,
        slice_size=640,
    )
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = enhanced_p.predict(img, threshold=0.35)
    assert len(dets) > 0
    assert hasattr(dets, "xyxy")
    assert hasattr(dets, "confidence")


def test_fuse_boxes_wbf_outlier_downweighting():
    """Verify WBF divides cluster confidence by total passes covering that region."""
    boxes = [[100.0, 100.0, 200.0, 300.0]]
    scores = [0.40]
    class_ids = [0]
    class_names = ["person"]

    # When total_passes=4 (e.g. 1 full + 1 TTA + 2 tiles), 1-off false positive is downweighted
    fused = fuse_boxes_wbf(boxes, scores, class_ids, class_names, total_passes=4)
    assert len(fused["boxes"]) == 1
    assert abs(fused["scores"][0] - 0.10) < 1e-4


def test_multiclass_gt_matching_indices(tmp_path: Path):
    """Verify evaluate_dataset_accuracy correctly aligns multi-class GT indices."""
    # Synthetic image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test_001.jpg"
    cv2.imwrite(str(img_path), img)

    # COCO JSON with 2 classes: person and head
    coco_dict = {
        "images": [{"id": 1, "file_name": "test_001.jpg", "width": 100, "height": 100}],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "head"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 80], "area": 4000},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [20, 10, 20, 20], "area": 400},
        ],
    }
    ann_path = tmp_path / "ann.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)

    # Mock predictor returning matching person and head
    class MockMultiPredictor:
        def predict(self, image_rgb, threshold=0.35):
            class _Dets:
                xyxy = np.array([[10, 10, 60, 90], [20, 10, 40, 30]], dtype=np.float32)
                confidence = np.array([0.90, 0.85], dtype=np.float32)
                class_id = np.array([1, 2], dtype=np.int32)
                data = {"class_name": ["person", "head"]}
            return _Dets()

    res = evaluate_dataset_accuracy(
        predictor=MockMultiPredictor(),
        test_images_dir=str(tmp_path),
        annotation_file=str(ann_path),
        target_classes=["person", "head"],
        iou_thresholds=[0.50],
        confidence_threshold=0.35,
    )
    assert res["mAP@50"] > 0.90
    assert res["recall@50"] == 1.0
    assert res["small_object_recall"] == 1.0


def test_run_predict_workflow(tmp_path: Path):
    """Verify run_predict executes and outputs annotated image and manifest."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(test_img_path), img)

    out_dir = tmp_path / "predict_out"

    # Test drawing helper
    annotated = draw_bounding_boxes(
        img,
        [{"box_xyxy": [10, 10, 50, 50], "confidence": 0.88, "class_name": "person"}],
    )
    assert annotated.shape == img.shape

    # Test run_predict with Mock
    class MockPredictor:
        def predict(self, frame_rgb, threshold=0.35):
            class _Dets:
                xyxy = np.array([[10, 10, 50, 50]], dtype=np.float32)
                confidence = np.array([0.88], dtype=np.float32)
                class_id = np.array([0], dtype=np.int32)
                data = {"class_name": ["person"]}
            return _Dets()

    import src.predict as predict_module
    orig_create_predictor = predict_module.create_predictor
    orig_create_model = predict_module.create_model
    try:
        predict_module.create_model = lambda **kwargs: None
        predict_module.create_predictor = lambda model: MockPredictor()

        res = run_predict(
            source=str(test_img_path),
            model_name="RFDETRLarge",
            output_dir=str(out_dir),
            save=True,
        )
        assert res["total_detections"] == 1
        assert (out_dir / "annotated_sample.jpg").is_file()
        assert (out_dir / "predictions_summary.json").is_file()
    finally:
        predict_module.create_predictor = orig_create_predictor
        predict_module.create_model = orig_create_model
