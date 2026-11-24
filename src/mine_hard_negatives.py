"""Hard-negative and low-confidence frame mining for video and camera feeds.

Identifies ambiguous detections (e.g., confidence between 0.15 and 0.45)
and potential false positives to curate high-impact samples for subsequent
retraining rounds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from src.config import HardMiningConfig, load_config
from src.model_factory import create_model, create_predictor

console = Console()


def extract_detections_data(detections_obj: Any) -> List[Dict[str, Any]]:
    """Uniformly unpack detections from rfdetr / supervision Detections object.
    
    Supports:
    - supervision.Detections (xyxy, confidence, class_id, data)
    - list of dicts or custom tuple formats
    """
    records: List[Dict[str, Any]] = []
    if detections_obj is None:
        return records

    # Check for supervision Detections style attributes
    if hasattr(detections_obj, "xyxy"):
        xyxy = detections_obj.xyxy
        conf = getattr(detections_obj, "confidence", None)
        class_id = getattr(detections_obj, "class_id", None)
        data = getattr(detections_obj, "data", {})
        class_names = data.get("class_name", None) if isinstance(data, dict) else None

        n = len(xyxy)
        for i in range(n):
            c_score = float(conf[i]) if conf is not None else 1.0
            cid = int(class_id[i]) if class_id is not None else 0
            cname = str(class_names[i]) if (class_names is not None and i < len(class_names)) else "person"
            box = xyxy[i].tolist() if hasattr(xyxy[i], "tolist") else list(xyxy[i])

            records.append({
                "box_xyxy": [float(x) for x in box],
                "confidence": c_score,
                "class_id": cid,
                "class_name": cname,
            })
    elif isinstance(detections_obj, (list, tuple)):
        for item in detections_obj:
            if isinstance(item, dict):
                records.append(item)
    return records


def draw_bounding_boxes(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 165, 255),  # Orange for ambiguous
) -> np.ndarray:
    """Draw bounding boxes and confidence labels on BGR image for human inspection."""
    vis = frame_bgr.copy()
    for d in detections:
        box = d["box_xyxy"]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        conf = d["confidence"]
        cname = d.get("class_name", "person")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cname}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def mine_hard_negatives(
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    input_frames_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    enhanced: bool = False,
    enable_tiling: Optional[bool] = None,
    enable_tta: Optional[bool] = None,
    config: Optional[HardMiningConfig] = None,
) -> Dict[str, Any]:
    """Mine hard negative and low-confidence frames from imagery.
    
    Args:
        model_name: RF-DETR model name (RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge).
        checkpoint_path: Optional custom fine-tuned checkpoint.
        input_frames_dir: Directory containing extracted frames.
        output_dir: Output directory for mined hard negatives.
        enhanced: Whether to use enhanced multi-pass inference for mining.
        enable_tiling: Override tiling mode.
        enable_tta: Override TTA mode.
        config: HardMiningConfig instance.
        
    Returns:
        Summary statistics dictionary.
    """
    cfg = load_config()
    m_name = model_name or cfg.model.name
    c_path = checkpoint_path or cfg.model.pretrain_weights
    mining_cfg = config or cfg.hard_mining

    in_dir = Path(input_frames_dir or mining_cfg.input_frames_dir)
    out_dir = Path(output_dir or mining_cfg.output_dir)

    images_out = out_dir / "images"
    vis_out = out_dir / "visualizations"
    images_out.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]Mining Hard Negatives for Retraining Round 2[/bold cyan]\n"
        f"Input directory: {in_dir}\n"
        f"Ambiguity threshold range: [{mining_cfg.min_confidence:.2f}, {mining_cfg.max_confidence:.2f}]\n"
        f"Enhanced Mode: [green]{'Enabled' if enhanced else 'Disabled'}[/green]\n"
        f"Destination: {out_dir}"
    )

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts] if in_dir.exists() else []

    if not frame_files:
        console.print(f"[yellow]No candidate images found in {in_dir}. Run frame extraction first.[/yellow]")
        return {"total_frames": 0, "mined_frames": 0, "ambiguous_detections": 0}

    # Initialize model and predictor
    model = create_model(model_name=m_name, pretrain_weights=c_path)
    predictor = create_predictor(model)

    if enhanced:
        from src.enhanced_inference import WorkZoneEnhancedPredictor
        console.print("[dim]Wrapping predictor with WorkZoneEnhancedPredictor for mining...[/dim]")
        predictor = WorkZoneEnhancedPredictor.from_config(base_predictor=predictor, config=cfg)
        if enable_tiling is not None:
            predictor.enable_tiling = enable_tiling
        if enable_tta is not None:
            predictor.enable_tta = enable_tta

    mined_records: List[Dict[str, Any]] = []
    total_ambiguous = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Mining frames", total=min(len(frame_files), mining_cfg.max_samples_per_video))

        for fpath in frame_files[:mining_cfg.max_samples_per_video]:
            progress.update(task, advance=1)
            frame_bgr = cv2.imread(str(fpath))
            if frame_bgr is None:
                continue

            # Convert OpenCV BGR frame to RGB for detector inference
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            try:
                # Predict with low threshold to capture ambiguous band
                raw_detections = predictor.predict(image_rgb, threshold=mining_cfg.min_confidence)
                detections = extract_detections_data(raw_detections)
            except Exception as e:
                console.print(f"[dim red]Inference error on {fpath.name}: {e}[/dim red]")
                continue

            # Filter for ambiguous detections within [min_confidence, max_confidence]
            ambiguous = [
                d for d in detections
                if mining_cfg.min_confidence <= d["confidence"] <= mining_cfg.max_confidence
            ]

            if ambiguous:
                total_ambiguous += len(ambiguous)
                # Save original image to hard negatives folder
                dest_img = images_out / fpath.name
                cv2.imwrite(str(dest_img), frame_bgr)

                # Save visualization overlay
                vis_img = draw_bounding_boxes(frame_bgr, ambiguous)
                dest_vis = vis_out / f"vis_{fpath.name}"
                cv2.imwrite(str(dest_vis), vis_img)

                mined_records.append({
                    "original_file": fpath.name,
                    "saved_file": dest_img.name,
                    "ambiguous_count": len(ambiguous),
                    "detections": ambiguous,
                })

    # Save manifest for retraining round 2
    manifest_path = out_dir / "mining_manifest.json"
    manifest_data = {
        "model_name": m_name,
        "checkpoint_used": c_path,
        "confidence_range": [mining_cfg.min_confidence, mining_cfg.max_confidence],
        "total_analyzed_frames": len(frame_files),
        "mined_frames_count": len(mined_records),
        "total_ambiguous_detections": total_ambiguous,
        "records": mined_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Display summary
    table = Table(title="Hard-Negative Mining Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Frames Analyzed", str(len(frame_files)))
    table.add_row("Frames with Ambiguous Detections", str(len(mined_records)))
    table.add_row("Total Ambiguous Instances", str(total_ambiguous))
    table.add_row("Confidence Window", f"[{mining_cfg.min_confidence:.2f}, {mining_cfg.max_confidence:.2f}]")
    table.add_row("Mining Manifest", str(manifest_path))

    console.print(table)
    return manifest_data


if __name__ == "__main__":
    mine_hard_negatives()
