"""WorkZone Vision CLI.

Commands:
  download          Download and verify RF-DETR model weights
  extract-frames    Extract sharp, non-duplicate CCTV frames
  validate-dataset  Verify COCO dataset integrity and class distributions
  train             Fine-tune model on custom CCTV dataset
  mine-negatives    Mine hard-negative and ambiguous CCTV frames
  evaluate          Benchmark Pretrained vs. Fine-Tuned (mAP@50, mAP@50:95, latency)
  export            Export checkpoint to ONNX format and verify
  pipeline          Run full end-to-end accuracy pipeline
"""

from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel

from src.config import PipelineConfig, load_config
from src.download_model import download_and_verify_model
from src.evaluate import benchmark_comparison
from src.export_model import export_to_onnx
from src.extract_frames import process_video_directory
from src.mine_hard_negatives import mine_hard_negatives
from src.train_finetune import train_model
from src.validate_dataset import validate_cctv_dataset

app = typer.Typer(
    name="workzone",
    help="WorkZone Vision: Edge-ready CCTV person detection fine-tuning and export pipeline.",
    add_completion=False,
)
console = Console()


@app.command("download")
def download_cmd(
    model_name: str = typer.Option("RFDETRSmall", "--model-name", "-m", help="RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Path to custom checkpoint"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Download and initialize Apache-2.0 RF-DETR model."""
    cfg = load_config(config)
    download_and_verify_model(model_name=model_name, pretrain_weights=weights, config=cfg)


@app.command("extract-frames")
def extract_cmd(
    video_dir: Optional[str] = typer.Option(None, "--video-dir", "-i", help="Directory of input CCTV video clips"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Destination directory for frames"),
    interval: Optional[int] = typer.Option(None, "--frame-interval", help="Sample every N frames"),
    blur_threshold: Optional[float] = typer.Option(None, "--blur-threshold", help="Variance of Laplacian threshold"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Extract sharp, non-duplicate frames from raw CCTV video clips."""
    cfg = load_config(config)
    if interval is not None:
        cfg.extraction.frame_interval = interval
    if blur_threshold is not None:
        cfg.extraction.blur_threshold = blur_threshold

    process_video_directory(
        video_dir=video_dir or cfg.extraction.raw_video_dir,
        output_dir=output_dir or cfg.extraction.extracted_frames_dir,
        config=cfg.extraction,
    )


@app.command("validate-dataset")
def validate_cmd(
    dataset_dir: Optional[str] = typer.Option(None, "--dataset-dir", "-d", help="Path to COCO dataset folder"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Comma-separated class names e.g. person,head"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Validate COCO format labeled dataset integrity, boxes, and class distribution."""
    cfg = load_config(config)
    target_classes = [c.strip() for c in classes.split(",")] if classes else cfg.detection.classes
    d_dir = dataset_dir or cfg.dataset.dataset_dir
    is_valid, _ = validate_cctv_dataset(dataset_dir=d_dir, config=cfg.dataset, target_classes=target_classes)
    if not is_valid:
        raise typer.Exit(code=1)


@app.command("train")
def train_cmd(
    model_name: Optional[str] = typer.Option(None, "--model-name", "-m", help="RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge"),
    dataset_dir: Optional[str] = typer.Option(None, "--dataset-dir", "-d", help="Path to COCO dataset directory"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Checkpoint output directory"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Training epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Checkpoint path to resume from"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Fine-tune RF-DETR on office CCTV data."""
    cfg = load_config(config)
    res = train_model(
        model_name=model_name or cfg.model.name,
        dataset_dir=dataset_dir or cfg.dataset.dataset_dir,
        output_dir=output_dir or cfg.training.output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resume_path=resume,
        config=cfg,
    )
    if not res.get("success", False):
        raise typer.Exit(code=1)


@app.command("mine-negatives")
def mine_cmd(
    input_dir: Optional[str] = typer.Option(None, "--input-dir", "-i", help="Directory of candidate CCTV frames"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory for mined hard negatives"),
    min_conf: Optional[float] = typer.Option(None, "--min-conf", help="Lower bound of ambiguous confidence"),
    max_conf: Optional[float] = typer.Option(None, "--max-conf", help="Upper bound of ambiguous confidence"),
    model_name: Optional[str] = typer.Option(None, "--model-name", "-m", help="Model name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Path to checkpoint weights"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Mine ambiguous, low-confidence CCTV frames for retraining round 2."""
    cfg = load_config(config)
    if min_conf is not None:
        cfg.hard_mining.min_confidence = min_conf
    if max_conf is not None:
        cfg.hard_mining.max_confidence = max_conf

    mine_hard_negatives(
        model_name=model_name or cfg.model.name,
        checkpoint_path=checkpoint,
        input_frames_dir=input_dir or cfg.hard_mining.input_frames_dir,
        output_dir=output_dir or cfg.hard_mining.output_dir,
        config=cfg.hard_mining,
    )


@app.command("evaluate")
def evaluate_cmd(
    model_name: Optional[str] = typer.Option(None, "--model-name", "-m", help="Model name"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-cp", help="Path to fine-tuned checkpoint"),
    test_dir: Optional[str] = typer.Option(None, "--test-dir", help="Directory with test images"),
    annotation_file: Optional[str] = typer.Option(None, "--annotation-file", help="COCO test JSON path"),
    report: Optional[str] = typer.Option(None, "--report", help="Output path for benchmark report"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Benchmark Pretrained vs. Fine-Tuned (mAP@50, mAP@50:95, latency ms)."""
    cfg = load_config(config)
    benchmark_comparison(
        model_name=model_name or cfg.model.name,
        checkpoint_path=checkpoint or cfg.export.checkpoint_path,
        test_images_dir=test_dir or cfg.evaluation.test_images_dir,
        annotation_file=annotation_file or cfg.evaluation.test_annotation_file,
        output_report_path=report,
        config=cfg,
    )


@app.command("export")
def export_cmd(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-cp", help="Path to checkpoint to export"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Target output directory"),
    model_name: Optional[str] = typer.Option(None, "--model-name", "-m", help="Model architecture"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Export fine-tuned RF-DETR model to ONNX format and verify runtime accuracy."""
    cfg = load_config(config)
    res = export_to_onnx(
        checkpoint_path=checkpoint or cfg.export.checkpoint_path,
        output_dir=output_dir or cfg.export.output_dir,
        model_name=model_name or cfg.model.name,
        config=cfg.export,
    )
    if not res.get("export_success", False):
        raise typer.Exit(code=1)


@app.command("pipeline")
def full_pipeline_cmd(
    model_name: str = typer.Option("RFDETRSmall", "--model-name", "-m", help="RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Execute complete end-to-end RF-DETR CCTV accuracy workflow."""
    cfg = load_config(config)
    cfg.model.name = model_name

    console.print(
        Panel(
            "[bold green]Starting End-to-End RF-DETR CCTV Accuracy Pipeline[/bold green]\n"
            "Stages:\n"
            " 1. Download & verify Apache-2.0 RF-DETR model\n"
            " 2. Extract & filter high-quality CCTV frames\n"
            " 3. Validate dataset integrity & annotations\n"
            " 4. Fine-tune model on custom CCTV data\n"
            " 5. Mine hard-negatives for round 2 retraining\n"
            " 6. Benchmark Pretrained vs. Fine-Tuned accuracy (mAP, latency)\n"
            " 7. Export and verify ONNX runtime model",
            title="Pipeline Coordinator",
            expand=False,
        )
    )

    # 1. Download
    console.print("\n[bold]Stage 1: Model Download[/bold]")
    download_and_verify_model(model_name=model_name, config=cfg)

    # 2. Extract frames
    console.print("\n[bold]Stage 2: Frame Extraction & Quality Filtering[/bold]")
    process_video_directory(config=cfg.extraction)

    # 3. Validate dataset
    console.print("\n[bold]Stage 3: Dataset Integrity Validation[/bold]")
    validate_cctv_dataset(config=cfg.dataset)

    # 4. Train
    console.print("\n[bold]Stage 4: Fine-Tuning[/bold]")
    train_summary = train_model(model_name=model_name, config=cfg)
    best_checkpoint = train_summary.get("best_checkpoint")

    # 5. Mine hard negatives
    console.print("\n[bold]Stage 5: Hard-Negative Mining[/bold]")
    mine_hard_negatives(model_name=model_name, checkpoint_path=best_checkpoint, config=cfg.hard_mining)

    # 6. Benchmark
    console.print("\n[bold]Stage 6: Pretrained vs. Fine-Tuned Accuracy Benchmarking[/bold]")
    benchmark_comparison(model_name=model_name, checkpoint_path=best_checkpoint, config=cfg)

    # 7. Export
    console.print("\n[bold]Stage 7: ONNX Export & Verification[/bold]")
    export_to_onnx(checkpoint_path=best_checkpoint, model_name=model_name, config=cfg.export)

    console.print("\n[bold green]✓ Full RF-DETR Accuracy Pipeline completed successfully![/bold green]")


if __name__ == "__main__":
    app()
