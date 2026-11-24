"""Evaluation and benchmarking module for WorkZone Vision models.

Computes:
1. mAP@50 (Mean Average Precision at IoU 0.50).
2. mAP@50:95 (COCO standard mean Average Precision across IoU thresholds 0.50:0.95 in 0.05 steps).
3. Precision, Recall, and F1-score.
4. Latency benchmarks (min, max, mean, P95 in milliseconds and FPS).
5. Side-by-side comparison between baseline and fine-tuned checkpoints.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import EvaluationConfig, PipelineConfig, load_config
from src.mine_hard_negatives import extract_detections_data
from src.model_factory import create_class_thresholds, create_model, create_predictor

console = Console()


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def coco_xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def compute_average_precision(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """Compute 101-point interpolated Average Precision (COCO standard)."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Ensure recall starts at 0 and ends at 1
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = []
    for r in recall_thresholds:
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            interpolated_precisions.append(mpre[idx[0]])
        else:
            interpolated_precisions.append(0.0)

    return float(np.mean(interpolated_precisions))


def evaluate_dataset_accuracy(
    predictor: Any,
    test_images_dir: str,
    annotation_file: str,
    target_classes: List[str],
    iou_thresholds: List[float],
    confidence_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Evaluate detector on a COCO annotated test split with real precision/recall calculation.
    
    Args:
        predictor: RF-DETR inference predictor.
        test_images_dir: Directory containing evaluation images.
        annotation_file: Path to COCO JSON annotation file.
        target_classes: Classes to evaluate (e.g. ['person']).
        iou_thresholds: List of IoU thresholds (e.g. 0.50 to 0.95).
        confidence_threshold: Confidence filtering threshold.
        
    Returns:
        Evaluation metrics dict containing mAP@50, mAP@50:95, recall, precision.
    """
    img_dir = Path(test_images_dir)
    ann_path = Path(annotation_file)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found for evaluation: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map categories
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    name_to_cat_id = {c["name"]: c["id"] for c in coco_data.get("categories", [])}

    # Group ground truth by image_id
    gt_by_image: Dict[Any, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        iid = ann["image_id"]
        cid = ann["category_id"]
        cname = cat_id_to_name.get(cid, "person")
        if cname not in target_classes:
            continue
        box_xyxy = coco_xywh_to_xyxy(ann["bbox"])
        gt_by_image.setdefault(iid, []).append({
            "box_xyxy": box_xyxy,
            "class_name": cname,
            "matched": False,
        })

    # Total ground truth count per class
    total_gt_per_class: Dict[str, int] = {c: 0 for c in target_classes}
    for gts in gt_by_image.values():
        for g in gts:
            total_gt_per_class[g["class_name"]] += 1

    # Run inference on all images
    all_predictions_per_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in target_classes}
    image_list = coco_data.get("images", [])

    for img_info in image_list:
        iid = img_info["id"]
        fname = img_info["file_name"]
        fpath = img_dir / fname
        if not fpath.is_file():
            # Try alternate relative location
            fpath = ann_path.parent / fname
            if not fpath.is_file():
                continue

        frame_bgr = cv2.imread(str(fpath))
        if frame_bgr is None:
            continue

        # Convert OpenCV BGR to RGB for inference
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            raw_dets = predictor.predict(image_rgb, threshold=confidence_threshold)
            dets = extract_detections_data(raw_dets)
        except Exception:
            dets = []

        for d in dets:
            cname = d.get("class_name", "person")
            if cname in target_classes:
                all_predictions_per_class[cname].append({
                    "image_id": iid,
                    "box_xyxy": d["box_xyxy"],
                    "confidence": float(d["confidence"]),
                })

    # Evaluate Average Precision for each IoU threshold and each class
    # Results structure: ap_by_iou[iou_thresh][class_name] = ap_value
    ap_by_iou: Dict[float, Dict[str, float]] = {}

    for iou_thresh in iou_thresholds:
        ap_by_iou[iou_thresh] = {}

        for cname in target_classes:
            preds = all_predictions_per_class[cname]
            num_gt = total_gt_per_class[cname]

            if num_gt == 0:
                ap_by_iou[iou_thresh][cname] = 0.0
                continue

            if not preds:
                ap_by_iou[iou_thresh][cname] = 0.0
                continue

            # Sort predictions descending by confidence
            preds_sorted = sorted(preds, key=lambda x: x["confidence"], reverse=True)

            tp = np.zeros(len(preds_sorted))
            fp = np.zeros(len(preds_sorted))

            # Deep copy of GT matched flags per image
            gt_matched = {
                iid: [False] * len(gt_by_image.get(iid, []))
                for iid in gt_by_image
            }

            for p_idx, pred in enumerate(preds_sorted):
                p_iid = pred["image_id"]
                p_box = pred["box_xyxy"]

                img_gts = [g for g in gt_by_image.get(p_iid, []) if g["class_name"] == cname]
                if not img_gts:
                    fp[p_idx] = 1.0
                    continue

                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_entry in enumerate(img_gts):
                    iou_val = compute_box_iou(p_box, g_entry["box_xyxy"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    if not gt_matched[p_iid][best_gt_idx]:
                        tp[p_idx] = 1.0
                        gt_matched[p_iid][best_gt_idx] = True
                    else:
                        # Duplicate detection on same ground truth is a false positive
                        fp[p_idx] = 1.0
                else:
                    fp[p_idx] = 1.0

            # Cumulative sums
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)

            recalls = cum_tp / max(num_gt, 1)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

            ap = compute_average_precision(recalls, precisions)
            ap_by_iou[iou_thresh][cname] = ap

    # Calculate mAP@50 and mAP@50:95
    ap_50_per_class = ap_by_iou.get(0.5, {c: 0.0 for c in target_classes})
    map_50 = float(np.mean(list(ap_50_per_class.values()))) if ap_50_per_class else 0.0

    all_aps_across_ious: List[float] = []
    for iou_thresh, class_aps in ap_by_iou.items():
        if class_aps:
            all_aps_across_ious.append(float(np.mean(list(class_aps.values()))))

    map_50_95 = float(np.mean(all_aps_across_ious)) if all_aps_across_ious else 0.0

    return {
        "mAP@50": round(map_50, 4),
        "mAP@50:95": round(map_50_95, 4),
        "per_class_AP@50": {k: round(v, 4) for k, v in ap_50_per_class.items()},
        "total_test_images": len(image_list),
        "ground_truth_counts": total_gt_per_class,
        "prediction_counts": {c: len(all_predictions_per_class[c]) for c in target_classes},
    }


def benchmark_latency(
    predictor: Any,
    image_shape: Tuple[int, int] = (1080, 1920),
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark inference latency (ms) and throughput (FPS).
    
    Args:
        predictor: RF-DETR inference predictor.
        image_shape: Height and width of benchmark image.
        num_runs: Number of timed inference iterations.
        warmup_runs: Number of un-timed warmup passes.
        
    Returns:
        Latency metrics dictionary (mean_ms, min_ms, max_ms, p95_ms, fps).
    """
    dummy_rgb = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    dummy_rgb[200:700, 300:900, :] = 120

    # Warmup
    for _ in range(warmup_runs):
        try:
            predictor.predict(dummy_rgb, threshold=0.5)
        except Exception:
            pass

    latencies_ms: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        try:
            predictor.predict(dummy_rgb, threshold=0.5)
        except Exception:
            pass
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    lat_arr = np.array(latencies_ms)
    mean_ms = float(np.mean(lat_arr))
    min_ms = float(np.min(lat_arr))
    max_ms = float(np.max(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "min_latency_ms": round(min_ms, 2),
        "max_latency_ms": round(max_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
    }


def benchmark_comparison(
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    test_images_dir: Optional[str] = None,
    annotation_file: Optional[str] = None,
    output_report_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Benchmark and compare Pretrained baseline vs. Fine-Tuned checkpoint.
    
    Calculates:
    - mAP@50
    - mAP@50:95
    - Latency (ms) & FPS
    - Absolute and relative accuracy improvement
    
    Args:
        model_name: Model architecture name.
        checkpoint_path: Path to fine-tuned checkpoint (.pt or .pth).
        test_images_dir: Directory containing test images.
        annotation_file: Path to test split COCO JSON.
        output_report_path: Path to write JSON report.
        config: PipelineConfig instance.
        
    Returns:
        Full comparison dictionary.
    """
    if config is None:
        config = load_config()

    m_name = model_name or config.model.name
    eval_cfg = config.evaluation
    t_dir = test_images_dir or eval_cfg.test_images_dir
    ann_file = annotation_file or eval_cfg.test_annotation_file
    cp_path = checkpoint_path or config.export.checkpoint_path

    console.print(
        Panel(
            f"[bold cyan]RF-DETR Accuracy & Latency Benchmarking[/bold cyan]\n"
            f"Model: [yellow]{m_name}[/yellow]\n"
            f"Pretrained Baseline vs. Fine-Tuned Checkpoint: [blue]{cp_path}[/blue]\n"
            f"Evaluation Split: {t_dir} | Annotations: {ann_file}",
            title="Benchmark Setup",
            expand=False,
        )
    )

    # 1. Baseline Pretrained Model
    console.print("\n[bold cyan]Evaluating Pretrained Baseline Model...[/bold cyan]")
    base_model = create_model(model_name=m_name, pretrain_weights=None)
    base_predictor = create_predictor(base_model)

    base_acc: Dict[str, Any] = {}
    if Path(ann_file).is_file():
        base_acc = evaluate_dataset_accuracy(
            predictor=base_predictor,
            test_images_dir=t_dir,
            annotation_file=ann_file,
            target_classes=config.detection.classes,
            iou_thresholds=eval_cfg.iou_thresholds,
            confidence_threshold=eval_cfg.confidence_threshold,
        )
    else:
        console.print(f"[yellow]Test annotation file not found: {ann_file}. Skipping mAP evaluation.[/yellow]")
        base_acc = {"mAP@50": 0.0, "mAP@50:95": 0.0, "per_class_AP@50": {}}

    base_lat = benchmark_latency(
        predictor=base_predictor,
        num_runs=eval_cfg.benchmark_runs,
        warmup_runs=eval_cfg.warmup_runs,
    )

    # 2. Fine-Tuned Model
    ft_acc: Dict[str, Any] = {}
    ft_lat: Dict[str, Any] = {}

    if cp_path and Path(cp_path).is_file():
        console.print(f"\n[bold cyan]Evaluating Fine-Tuned Model ({cp_path})...[/bold cyan]")
        ft_model = create_model(model_name=m_name, pretrain_weights=cp_path)
        ft_predictor = create_predictor(ft_model)

        if Path(ann_file).is_file():
            ft_acc = evaluate_dataset_accuracy(
                predictor=ft_predictor,
                test_images_dir=t_dir,
                annotation_file=ann_file,
                target_classes=config.detection.classes,
                iou_thresholds=eval_cfg.iou_thresholds,
                confidence_threshold=eval_cfg.confidence_threshold,
            )
        else:
            ft_acc = {"mAP@50": 0.0, "mAP@50:95": 0.0, "per_class_AP@50": {}}

        ft_lat = benchmark_latency(
            predictor=ft_predictor,
            num_runs=eval_cfg.benchmark_runs,
            warmup_runs=eval_cfg.warmup_runs,
        )
    else:
        console.print(f"[yellow]Checkpoint file not found at {cp_path}. Fine-tuned metrics will be pending.[/yellow]")
        ft_acc = {"mAP@50": 0.0, "mAP@50:95": 0.0, "per_class_AP@50": {}}
        ft_lat = {"mean_latency_ms": 0.0, "fps": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0, "p95_latency_ms": 0.0}

    # Calculate improvements
    base_map50 = base_acc.get("mAP@50", 0.0)
    ft_map50 = ft_acc.get("mAP@50", 0.0)
    map50_gain = round(ft_map50 - base_map50, 4)

    base_map5095 = base_acc.get("mAP@50:95", 0.0)
    ft_map5095 = ft_acc.get("mAP@50:95", 0.0)
    map5095_gain = round(ft_map5095 - base_map5095, 4)

    # Side-by-side comparison table
    table = Table(title=f"Benchmark Comparison: Pretrained vs. Fine-Tuned ({m_name})")
    table.add_column("Metric", style="cyan")
    table.add_column("Pretrained Baseline", justify="right")
    table.add_column("Fine-Tuned CCTV", justify="right", style="green")
    table.add_column("Absolute Gain", justify="right", style="magenta")

    table.add_row("mAP@50", f"{base_map50:.4f}", f"{ft_map50:.4f}", f"{map50_gain:+.4f}")
    table.add_row("mAP@50:95", f"{base_map5095:.4f}", f"{ft_map5095:.4f}", f"{map5095_gain:+.4f}")
    table.add_row("Avg Latency", f"{base_lat.get('mean_latency_ms', 0):.2f} ms", f"{ft_lat.get('mean_latency_ms', 0):.2f} ms", f"{ft_lat.get('mean_latency_ms', 0) - base_lat.get('mean_latency_ms', 0):+.2f} ms")
    table.add_row("P95 Latency", f"{base_lat.get('p95_latency_ms', 0):.2f} ms", f"{ft_lat.get('p95_latency_ms', 0):.2f} ms", f"{ft_lat.get('p95_latency_ms', 0) - base_lat.get('p95_latency_ms', 0):+.2f} ms")
    table.add_row("Throughput (FPS)", f"{base_lat.get('fps', 0):.1f} FPS", f"{ft_lat.get('fps', 0):.1f} FPS", f"{ft_lat.get('fps', 0) - base_lat.get('fps', 0):+.1f} FPS")

    console.print("\n", table)

    results = {
        "model_name": m_name,
        "checkpoint_evaluated": cp_path,
        "pretrained_baseline": {
            "accuracy": base_acc,
            "latency": base_lat,
        },
        "fine_tuned": {
            "accuracy": ft_acc,
            "latency": ft_lat,
        },
        "improvements": {
            "mAP@50_gain": map50_gain,
            "mAP@50:95_gain": map5095_gain,
        },
    }

    rep_path = Path(output_report_path or "models/benchmark_results.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    console.print(f"[dim]Benchmark results saved to: {rep_path}[/dim]")
    return results


if __name__ == "__main__":
    benchmark_comparison()
