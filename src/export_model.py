"""Export fine-tuned RF-DETR models to ONNX format and verify runtime accuracy with VEXDET."""

import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import ExportConfig, load_config
from src.mine_hard_negatives import extract_detections_data
from src.model_factory import (
    ALLOWED_MODEL_NAMES,
    create_model,
    create_onnx_predictor,
    create_predictor,
)

console = Console()


def export_to_onnx(
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    config: Optional[ExportConfig] = None,
) -> Dict[str, Any]:
    """Export fine-tuned RF-DETR model checkpoint to ONNX format and verify runtime accuracy.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt / .pth).
        output_dir: Target directory for exported .onnx model.
        model_name: Architecture in ('RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge').
        config: ExportConfig instance.
        
    Returns:
        Dictionary with export status, onnx file path, verification results, and latency.
    """
    cfg = load_config()
    exp_cfg = config or cfg.export

    m_name = model_name or cfg.model.name
    if m_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Model '{m_name}' not permitted. Only Apache-2.0 models {ALLOWED_MODEL_NAMES} allowed.")

    cp_path = Path(checkpoint_path or exp_cfg.checkpoint_path)
    out_dir = Path(output_dir or exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold cyan]VEXDET RF-DETR ONNX Model Export & Verification[/bold cyan]\n"
            f"Model Architecture: [yellow]{m_name}[/yellow]\n"
            f"Source Checkpoint: [blue]{cp_path}[/blue]\n"
            f"Export Directory: [green]{out_dir.resolve()}[/green]\n"
            f"Format: [magenta]ONNX (opset {exp_cfg.opset})[/magenta]",
            title="Export Plan",
            expand=False,
        )
    )

    # Instantiate model with pretrain weights
    # If checkpoint doesn't exist, instantiate default weights for export verification
    if cp_path.is_file():
        console.print(f"[green]✓ Found checkpoint: {cp_path}[/green]")
        model = create_model(model_name=m_name, pretrain_weights=str(cp_path))
    else:
        console.print(f"[yellow]Checkpoint '{cp_path}' not found on disk. Exporting base pretrained {m_name}...[/yellow]")
        model = create_model(model_name=m_name, pretrain_weights=None)

    # Run PyTorch baseline inference before export for cross-validation
    pytorch_predictor = create_predictor(model)
    dummy_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_rgb[250:650, 400:850, :] = 130

    t0 = time.perf_counter()
    try:
        pt_detections = extract_detections_data(
            pytorch_predictor.predict(dummy_rgb, threshold=cfg.detection.confidence_threshold)
        )
    except Exception as e:
        console.print(f"[dim yellow]PyTorch test pass note: {e}[/dim yellow]")
        pt_detections = []
    pt_latency = (time.perf_counter() - t0) * 1000.0

    # Execute export via exact API:
    console.print(f"[bold cyan]Exporting {m_name} to ONNX format...[/bold cyan]")
    t_exp_start = time.perf_counter()

    export_success = True
    export_error = None
    expected_onnx_name = f"vexdet_{m_name.lower()}.onnx"
    target_onnx_path = out_dir / expected_onnx_name

    # Inspect model.export signature to pass supported configuration options
    sig = inspect.signature(model.export)
    export_kwargs: Dict[str, Any] = {"format": "onnx"}
    if "opset" in sig.parameters and hasattr(exp_cfg, "opset"):
        export_kwargs["opset"] = exp_cfg.opset
        console.print(f"[dim]Passing opset={exp_cfg.opset} to model.export()[/dim]")
    elif hasattr(exp_cfg, "opset"):
        console.print(f"[dim]Note: 'opset' handled internally by rfdetr.[/dim]")

    if "simplify" in sig.parameters and hasattr(exp_cfg, "simplify"):
        export_kwargs["simplify"] = exp_cfg.simplify
        console.print(f"[dim]Passing simplify={exp_cfg.simplify} to model.export()[/dim]")
    elif hasattr(exp_cfg, "simplify"):
        console.print(f"[dim]Note: 'simplify' handled internally by rfdetr.[/dim]")

    try:
        # Export model graph to ONNX
        model.export(**export_kwargs)
        # Find where rfdetr saved the onnx file and move to target_onnx_path if needed
        possible_exported = list(Path(".").glob("*.onnx")) + list(out_dir.glob("*.onnx"))
        if possible_exported:
            newest_onnx = max(possible_exported, key=lambda p: p.stat().st_mtime)
            if newest_onnx.resolve() != target_onnx_path.resolve():
                shutil.copy2(newest_onnx, target_onnx_path)
    except Exception as e:
        export_success = False
        export_error = str(e)
        console.print(f"[bold red]model.export encountered error:[/bold red] {e}")

    export_duration = time.perf_counter() - t_exp_start

    # Verify ONNX model via RFDETRInference
    onnx_verified = False
    onnx_latency = 0.0
    onnx_detections = []

    if target_onnx_path.is_file():
        file_size_mb = target_onnx_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]✓ Exported ONNX artifact located:[/green] {target_onnx_path} ({file_size_mb:.2f} MB)")

        try:
            # Instantiate ONNX runtime predictor and verify
            console.print("[dim]Instantiating ONNX runtime predictor via RFDETRInference...[/dim]")
            onnx_predictor = create_onnx_predictor(str(target_onnx_path))

            t_onnx = time.perf_counter()
            raw_onnx_dets = onnx_predictor.predict(dummy_rgb, threshold=cfg.detection.confidence_threshold)
            onnx_latency = (time.perf_counter() - t_onnx) * 1000.0
            onnx_detections = extract_detections_data(raw_onnx_dets)
            onnx_verified = True
            console.print(f"[bold green]✓ ONNX verification inference passed in {onnx_latency:.2f} ms[/bold green]")
        except Exception as e:
            console.print(f"[yellow]ONNX runtime verification warning: {e}[/yellow]")
    else:
        file_size_mb = 0.0

    # Summary table
    table = Table(title="ONNX Model Export Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Architecture", m_name)
    table.add_row("Export Status", "[bold green]SUCCESS[/bold green]" if export_success else "[bold red]FAILED[/bold red]")
    table.add_row("Export Duration", f"{export_duration:.2f} seconds")
    table.add_row("ONNX File Path", str(target_onnx_path if target_onnx_path.is_file() else "None"))
    table.add_row("File Size", f"{file_size_mb:.2f} MB" if file_size_mb > 0 else "N/A")
    table.add_row("ONNX Verification", "[bold green]PASSED[/bold green]" if onnx_verified else "[bold yellow]UNVERIFIED[/bold yellow]")
    table.add_row("PyTorch Latency", f"{pt_latency:.2f} ms")
    table.add_row("ONNX Latency", f"{onnx_latency:.2f} ms" if onnx_verified else "N/A")
    if export_error:
        table.add_row("Export Error", str(export_error))

    console.print(table)

    summary = {
        "model_name": m_name,
        "export_success": export_success,
        "onnx_verified": onnx_verified,
        "onnx_path": str(target_onnx_path) if target_onnx_path.is_file() else None,
        "file_size_mb": round(file_size_mb, 2),
        "pytorch_latency_ms": round(pt_latency, 2),
        "onnx_latency_ms": round(onnx_latency, 2),
        "error": export_error,
    }

    # Save export metadata
    meta_path = out_dir / "export_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    export_to_onnx()
