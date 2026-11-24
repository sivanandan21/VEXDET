"""Fine-tuning module for WorkZone Vision RF-DETR models.

Supports training on custom CCTV datasets for single-class person detection
or dual-class person and head detection.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PipelineConfig, TrainingConfig, load_config
from src.model_factory import ALLOWED_MODEL_NAMES, create_model, is_cuda_available
from src.validate_dataset import validate_cctv_dataset

console = Console()


def train_model(
    model_name: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum_steps: Optional[int] = None,
    lr: Optional[float] = None,
    early_stopping: Optional[int] = None,
    resume_path: Optional[str] = None,
    pretrain_weights: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Fine-tune an Apache-2.0 RF-DETR model on custom office CCTV dataset.
    
    Args:
        model_name: Architecture in ('RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge').
        dataset_dir: Path to root dataset directory containing train/val splits.
        output_dir: Directory where checkpoints and training logs are saved.
        epochs: Number of training epochs.
        batch_size: Batch size per step.
        grad_accum_steps: Gradient accumulation steps.
        lr: Learning rate.
        early_stopping: Epoch patience for early stopping.
        resume_path: Optional checkpoint path to resume training from.
        pretrain_weights: Optional initial checkpoint weights.
        config: PipelineConfig instance.
        
    Returns:
        Dictionary containing training metadata, checkpoint path, and status.
    """
    if config is None:
        config = load_config()

    m_name = model_name or config.model.name
    if m_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Model '{m_name}' not permitted. Only Apache-2.0 models {ALLOWED_MODEL_NAMES} allowed.")

    t_cfg = config.training
    d_dir = Path(dataset_dir or config.dataset.dataset_dir)
    out_dir = Path(output_dir or t_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = epochs if epochs is not None else t_cfg.epochs
    bs = batch_size if batch_size is not None else t_cfg.batch_size
    gas = grad_accum_steps if grad_accum_steps is not None else t_cfg.grad_accum_steps
    learn_rate = lr if lr is not None else t_cfg.lr
    es = early_stopping if early_stopping is not None else t_cfg.early_stopping
    res_path = resume_path or t_cfg.resume_path
    init_weights = pretrain_weights or config.model.pretrain_weights

    # Hardware detection
    cuda_active = is_cuda_available()
    device_name = "CUDA (GPU)" if cuda_active else "CPU"
    if not cuda_active:
        console.print(
            "[yellow]Warning: No CUDA GPU detected. Running fine-tuning on CPU. "
            "Batch size adjusted if necessary to prevent out-of-memory.[/yellow]"
        )
        if bs > 2:
            bs = 2

    console.print(
        Panel(
            f"[bold cyan]RF-DETR Object Detection Fine-Tuning Pipeline[/bold cyan]\n"
            f"Model: [yellow]{m_name}[/yellow]\n"
            f"Dataset: [green]{d_dir.resolve()}[/green]\n"
            f"Classes: [green]{config.detection.classes}[/green]\n"
            f"Device: [magenta]{device_name}[/magenta]\n"
            f"Epochs: {n_epochs} | Batch Size: {bs} | Grad Accum: {gas} | LR: {learn_rate}\n"
            f"Output Checkpoints: [blue]{out_dir.resolve()}[/blue]\n"
            f"Resume Checkpoint: {res_path or 'None (From pre-trained)'}",
            title="Training Parameters",
            expand=False,
        )
    )

    # Validate dataset before triggering expensive training loop
    if d_dir.exists():
        console.print("[dim]Verifying dataset format prior to training...[/dim]")
        is_valid, val_report = validate_cctv_dataset(str(d_dir), config=config.dataset)
        if not is_valid:
            console.print(
                "[bold yellow]Dataset validation flagged errors or missing files. "
                "Review dataset_validation_report.json before training.[/bold yellow]"
            )

    # Instantiate model
    model = create_model(model_name=m_name, pretrain_weights=init_weights)

    console.print(f"[bold green]Starting fine-tuning for {n_epochs} epochs...[/bold green]")
    t_start = time.perf_counter()

    training_succeeded = True
    error_message = None

    try:
        # Launch training loop
        model.train(
            dataset_dir=str(d_dir),
            epochs=n_epochs,
            batch_size=bs,
            grad_accum_steps=gas,
            lr=learn_rate,
            output_dir=str(out_dir),
            early_stopping=es,
            resume=str(res_path) if res_path else None,
        )
    except Exception as e:
        training_succeeded = False
        error_message = str(e)
        console.print(f"[bold red]Training execution terminated with error:[/bold red] {e}")

    total_training_duration = time.perf_counter() - t_start

    # Locate best checkpoint produced
    best_checkpoint = None
    candidate_checkpoints = [
        out_dir / "best.pt",
        out_dir / "checkpoint_best_total.pth",
        out_dir / "checkpoint_best_regular.pth",
        out_dir / "best.pth",
        out_dir / "last.pt",
    ]
    for cp in candidate_checkpoints:
        if cp.is_file():
            best_checkpoint = str(cp)
            break

    # If no checkpoint found directly, search out_dir recursively
    if not best_checkpoint and out_dir.exists():
        pts = list(out_dir.glob("*.pt")) + list(out_dir.glob("*.pth"))
        if pts:
            best_checkpoint = str(pts[0])

    # Summary table
    table = Table(title="Fine-Tuning Execution Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Result", style="green")

    table.add_row("Model Architecture", m_name)
    table.add_row("Training Status", "[bold green]COMPLETED[/bold green]" if training_succeeded else "[bold red]FAILED[/bold red]")
    table.add_row("Duration", f"{total_training_duration:.2f} seconds ({total_training_duration / 60.0:.2f} mins)")
    table.add_row("Target Epochs", str(n_epochs))
    table.add_row("Batch Size", str(bs))
    table.add_row("Best Checkpoint", str(best_checkpoint or "None located"))
    if error_message:
        table.add_row("Error", str(error_message))

    console.print(table)

    summary = {
        "model_name": m_name,
        "success": training_succeeded,
        "duration_seconds": total_training_duration,
        "epochs": n_epochs,
        "batch_size": bs,
        "learning_rate": learn_rate,
        "dataset_dir": str(d_dir),
        "output_dir": str(out_dir),
        "best_checkpoint": best_checkpoint,
        "error": error_message,
    }

    # Save training metadata summary
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_model()
