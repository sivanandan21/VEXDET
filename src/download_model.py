"""Model download and initialization routines for VEXDET (Vision Extraction Detector).

Downloads official weights and verifies inference execution
across CPU and GPU environments.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PipelineConfig, load_config
from src.model_factory import (
    ALLOWED_MODEL_NAMES,
    create_class_thresholds,
    create_model,
    create_predictor,
    is_cuda_available,
)

console = Console()


def download_and_verify_model(
    model_name: str = "RFDETRSmall",
    pretrain_weights: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Download and verify an Apache-2.0 RF-DETR model.
    
    Args:
        model_name: Must be one of RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge.
        pretrain_weights: Optional path to custom checkpoint.
        config: Optional PipelineConfig instance.
        
    Returns:
        Dictionary containing model metadata, inference verification status, and timing.
    """
    if config is None:
        config = load_config()

    if model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(
            f"Model '{model_name}' is not allowed. Only Apache-2.0 models "
            f"{ALLOWED_MODEL_NAMES} are permitted."
        )

    cuda_available = is_cuda_available()
    device_mode = "CUDA" if (cuda_available and config.model.device != "cpu") else "CPU"

    console.print(
        Panel(
            f"[bold cyan]RF-DETR Apache-2.0 Model Downloader[/bold cyan]\n"
            f"Target Model: [yellow]{model_name}[/yellow]\n"
            f"Pretrained Weights: [yellow]{pretrain_weights or 'Standard Official (Apache-2.0)'}[/yellow]\n"
            f"Execution Device: [magenta]{device_mode}[/magenta]\n"
            f"License: [green]Apache-2.0 (Commercial & Research Permitted)[/green]",
            title="Download Plan",
            expand=False,
        )
    )

    t0 = time.perf_counter()
    model = create_model(model_name=model_name, pretrain_weights=pretrain_weights)
    download_time = time.perf_counter() - t0

    # Initialize predictor (NEVER inplace=True)
    t1 = time.perf_counter()
    predictor = create_predictor(model)
    init_time = time.perf_counter() - t1

    # Run verification inference pass on synthetic RGB frame (simulating 1080p camera frame)
    console.print("[dim]Running verification inference on synthetic 1080p RGB frame...[/dim]")
    dummy_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add dummy pattern
    dummy_rgb[200:600, 400:800, :] = 128

    thresholds = create_class_thresholds(
        config.detection.class_thresholds,
        default_threshold=config.detection.default_threshold,
    )

    t2 = time.perf_counter()
    try:
        detections = predictor.predict(dummy_rgb, threshold=thresholds)
        test_pass = True
        error_msg = None
    except Exception as e:
        test_pass = False
        error_msg = str(e)
        detections = None
        console.print(f"[bold red]Inference test warning:[/bold red] {e}")

    inference_time = time.perf_counter() - t2

    # Verification summary table
    table = Table(title="Model Download & Verification Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model Architecture", model_name)
    table.add_row("License", "Apache-2.0 Verified")
    table.add_row("Device Mode", device_mode)
    table.add_row("Instantiation Time", f"{download_time:.3f} s")
    table.add_row("Predictor Init Time", f"{init_time:.3f} s")
    table.add_row("Verification Inference Time", f"{inference_time * 1000.0:.2f} ms")
    table.add_row("Verification Status", "[bold green]PASSED[/bold green]" if test_pass else "[bold red]FAILED[/bold red]")
    if error_msg:
        table.add_row("Error Details", str(error_msg))

    console.print(table)

    return {
        "model_name": model_name,
        "license": "Apache-2.0",
        "device": device_mode,
        "instantiation_time_s": download_time,
        "predictor_init_time_s": init_time,
        "verification_latency_ms": inference_time * 1000.0,
        "verification_passed": test_pass,
        "model_instance": model,
        "predictor_instance": predictor,
    }


if __name__ == "__main__":
    download_and_verify_model()
