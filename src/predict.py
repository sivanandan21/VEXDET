"""Inference and visualization module for VEXDET.

Supports:
1. Single image, directory of images, and video feeds.
2. Standard PyTorch, ONNX runtime, and VexdetEnhancedPredictor modes.
3. Bounding box overlay rendering and prediction summary exporting.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PipelineConfig, load_config
from src.enhanced_inference import VexdetEnhancedPredictor, unpack_detections
from src.model_factory import (
    ALLOWED_MODEL_NAMES,
    create_class_thresholds,
    create_model,
    create_onnx_predictor,
    create_predictor,
)

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def draw_bounding_boxes(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    box_color: tuple = (0, 220, 110),
    text_color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Render bounding boxes and label tags onto a BGR OpenCV image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        box = det.get("box_xyxy", [0, 0, 0, 0])
        conf = float(det.get("confidence", 1.0))
        cname = str(det.get("class_name", "object"))

        x1 = int(np.clip(box[0], 0, w - 1))
        y1 = int(np.clip(box[1], 0, h - 1))
        x2 = int(np.clip(box[2], 0, w - 1))
        y2 = int(np.clip(box[3], 0, h - 1))

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw text label banner
        label = f"{cname} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        bg_y1 = max(0, y1 - label_h - baseline - 4)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(w - 1, x1 + label_w + 6)

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (bg_x1 + 3, bg_y2 - baseline - 1),
            font,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )

    return annotated


def run_predict(
    source: str,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    threshold: Optional[float] = None,
    enhanced: bool = False,
    enable_tiling: Optional[bool] = None,
    enable_tta: Optional[bool] = None,
    enable_clahe: Optional[bool] = None,
    output_dir: Optional[str] = "runs/predict",
    save: bool = True,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Execute prediction on image(s) or video source and produce annotated output.
    
    Args:
        source: File path, directory path, or webcam ID ('0').
        model_name: Model architecture name (e.g. 'RFDETRLarge').
        checkpoint_path: Optional PyTorch checkpoint path.
        onnx_path: Optional ONNX runtime model path.
        threshold: Confidence threshold float.
        enhanced: Enable VexdetEnhancedPredictor (CLAHE + Slicing + TTA + WBF).
        enable_tiling: Toggle sliced/tiled inference mode.
        enable_tta: Toggle test-time augmentation.
        enable_clahe: Toggle CLAHE preprocessing.
        output_dir: Directory where annotated results and summaries are saved.
        save: Whether to save annotated images/videos to disk.
        config: PipelineConfig instance.
        
    Returns:
        Dictionary containing prediction results, count of detections, and timing.
    """
    if config is None:
        config = load_config()

    m_name = model_name or config.model.name
    if m_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Model '{m_name}' not permitted. Allowed models: {ALLOWED_MODEL_NAMES}")

    conf_thresh = threshold if threshold is not None else config.detection.confidence_threshold
    out_path = Path(output_dir or "runs/predict")
    if save:
        out_path.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Predictor
    runtime_mode = "PyTorch"
    if onnx_path and Path(onnx_path).is_file():
        console.print(f"[bold cyan]Loading ONNX runtime model:[/bold cyan] {onnx_path}")
        base_predictor = create_onnx_predictor(str(onnx_path))
        runtime_mode = f"ONNX Runtime ({Path(onnx_path).name})"
    else:
        cp = checkpoint_path or (config.export.checkpoint_path if Path(config.export.checkpoint_path).is_file() else None)
        if cp and Path(cp).is_file():
            console.print(f"[bold cyan]Loading fine-tuned checkpoint:[/bold cyan] {cp}")
            model = create_model(model_name=m_name, pretrain_weights=str(cp))
            runtime_mode = f"PyTorch Fine-Tuned ({Path(cp).name})"
        else:
            console.print(f"[bold cyan]Loading pretrained baseline weights for:[/bold cyan] {m_name}")
            model = create_model(model_name=m_name, pretrain_weights=None)
            runtime_mode = f"PyTorch Pretrained ({m_name})"
        base_predictor = create_predictor(model)

    if enhanced:
        console.print("[green]Wrapping with VexdetEnhancedPredictor (CLAHE + Slicing + TTA + WBF)...[/green]")
        predictor = VexdetEnhancedPredictor.from_config(base_predictor=base_predictor, config=config, force_enhanced=True)
        if enable_tiling is not None:
            predictor.enable_tiling = enable_tiling
        if enable_tta is not None:
            predictor.enable_tta = enable_tta
        if enable_clahe is not None:
            predictor.enable_clahe = enable_clahe
    else:
        predictor = base_predictor

    # 2. Determine Source Type
    src_path = Path(source)
    is_webcam = source.isdigit()
    image_paths: List[Path] = []
    is_video = False

    if is_webcam:
        cam_idx = int(source)
    elif src_path.is_file():
        suffix = src_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_paths = [src_path]
        elif suffix in VIDEO_EXTENSIONS:
            is_video = True
        else:
            raise ValueError(f"Unsupported file format: {src_path.suffix}")
    elif src_path.is_dir():
        image_paths = sorted([
            f for f in src_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in directory: {src_path}")
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    console.print(
        Panel(
            f"[bold cyan]VEXDET Vision Predictor[/bold cyan]\n"
            f"Source: [yellow]{source}[/yellow]\n"
            f"Runtime: [blue]{runtime_mode}[/blue]\n"
            f"Threshold: [magenta]{conf_thresh:.2f}[/magenta]\n"
            f"Engine: [green]{'Enhanced (Multi-Pass WBF)' if enhanced else 'Standard Single-Pass'}[/green]\n"
            f"Output: [dim]{out_path.resolve()}[/dim]",
            title="Inference Session",
            expand=False,
        )
    )

    all_results: List[Dict[str, Any]] = []
    total_detections = 0
    latencies: List[float] = []

    # 3. Process Video or Images
    if is_video or is_webcam:
        cap = cv2.VideoCapture(int(source) if is_webcam else str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save and not is_webcam:
            out_vid_path = out_path / f"annotated_{src_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (frame_w, frame_h))

        frame_idx = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                raw_dets = predictor.predict(frame_rgb, threshold=conf_thresh)
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)

                dets = unpack_detections(raw_dets)
                total_detections += len(dets)

                annotated = draw_bounding_boxes(frame_bgr, dets)
                if writer is not None:
                    writer.write(annotated)

                if frame_idx % 30 == 0:
                    console.print(f"[dim]Frame {frame_idx}: {len(dets)} detections ({lat:.1f} ms)[/dim]")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                console.print(f"[green]✓ Annotated video saved to:[/green] {out_vid_path}")

    else:
        # Process individual image(s)
        for p in image_paths:
            frame_bgr = cv2.imread(str(p))
            if frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            raw_dets = predictor.predict(frame_rgb, threshold=conf_thresh)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            dets = unpack_detections(raw_dets)
            total_detections += len(dets)

            if save:
                annotated = draw_bounding_boxes(frame_bgr, dets)
                save_file = out_path / f"annotated_{p.name}"
                cv2.imwrite(str(save_file), annotated)

            all_results.append({
                "file": p.name,
                "latency_ms": round(lat, 2),
                "detections_count": len(dets),
                "detections": dets,
            })

    # Summary Display
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    table = Table(title="VEXDET Inference Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source", str(source))
    table.add_row("Runtime", runtime_mode)
    table.add_row("Frames / Images Processed", str(len(latencies)))
    table.add_row("Total Objects Detected", str(total_detections))
    table.add_row("Mean Latency", f"{avg_latency:.2f} ms")
    table.add_row("Throughput", f"{(1000.0 / avg_latency):.1f} FPS" if avg_latency > 0 else "N/A")
    table.add_row("Results Saved To", str(out_path.resolve()) if save else "None (save=False)")

    console.print("\n", table)

    summary_data = {
        "source": str(source),
        "runtime_mode": runtime_mode,
        "total_frames": len(latencies),
        "total_detections": total_detections,
        "mean_latency_ms": round(avg_latency, 2),
        "results": all_results,
    }

    if save and all_results:
        summary_json = out_path / "predictions_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        console.print(f"[dim]Saved predictions manifest to: {summary_json}[/dim]")

    return summary_data
