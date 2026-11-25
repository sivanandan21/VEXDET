"""Model factory for VEXDET RF-DETR architectures.

Provides factory helpers for model instantiation, inference predictor creation,
and ONNX runtime initialization across CPU and CUDA backends.
"""

from typing import Any, Dict, Optional, Type
import rich
from rich.console import Console

console = Console()

# Permitted Apache-2.0 model registry
ALLOWED_MODEL_NAMES = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")


def is_cuda_available() -> bool:
    """Check if CUDA is available via torch without throwing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_rfdetr_class(model_name: str) -> Type[Any]:
    """Dynamically import and return the requested Apache-2.0 RF-DETR model class.
    
    Args:
        model_name: One of 'RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge'.
        
    Returns:
        The RF-DETR model class.
        
    Raises:
        ValueError: If model_name is not in the strict Apache-2.0 whitelist.
        ImportError: If rfdetr is not installed.
    """
    if model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(
            f"Disallowed model architecture '{model_name}'. Allowed Apache-2.0 models: {ALLOWED_MODEL_NAMES}."
        )

    try:
        import rfdetr
    except ImportError as e:
        raise ImportError(
            "rfdetr is not installed. Install via: pip install 'rfdetr[train,onnx,stream]>=1.12.0'"
        ) from e

    if not hasattr(rfdetr, model_name):
        raise AttributeError(f"rfdetr package does not export '{model_name}'.")

    return getattr(rfdetr, model_name)


def create_model(
    model_name: str = "RFDETRSmall",
    pretrain_weights: Optional[str] = None,
) -> Any:
    """Instantiate an Apache-2.0 RF-DETR model.
    
    Args:
        model_name: Name of model in ('RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge').
        pretrain_weights: Path to checkpoint weights if fine-tuned, or None for base weights.
        
    Returns:
        Instantiated RF-DETR model.
    """
    model_cls = get_rfdetr_class(model_name)
    console.print(
        f"[bold green]✓[/bold green] Instantiating Apache-2.0 model: [cyan]{model_name}[/cyan] "
        f"(weights: {pretrain_weights or 'default pretrained'})"
    )

    if pretrain_weights is not None:
        return model_cls(pretrain_weights=str(pretrain_weights))
    return model_cls()


def create_predictor(model: Any) -> Any:
    """Create inference predictor from an instantiated RF-DETR model.
    
    Args:
        model: Instantiated RF-DETR model.
        
    Returns:
        Inference predictor instance.
    """
    console.print("[dim]Creating RF-DETR inference predictor...[/dim]")
    return model.inference()


def create_onnx_predictor(onnx_path: str) -> Any:
    """Create inference predictor from exported ONNX model.
    
    Args:
        onnx_path: Path to exported .onnx file.
        
    Returns:
        RFDETRInference predictor instance.
    """
    try:
        from rfdetr import RFDETRInference
    except ImportError as e:
        raise ImportError("rfdetr is not installed. Please install rfdetr>=1.12.0") from e

    console.print(f"[bold green]✓[/bold green] Loading ONNX predictor from: [cyan]{onnx_path}[/cyan]")
    return RFDETRInference(str(onnx_path))


def create_class_thresholds(
    thresholds: Optional[Dict[str, float]] = None,
    default_threshold: float = 0.50,
) -> Any:
    """Create ClassThresholds for fine-grained class-level filtering.
    
    Args:
        thresholds: Dictionary of class name to threshold (e.g. {"person": 0.35}).
        default_threshold: Fallback threshold for non-specified classes.
        
    Returns:
        rfdetr.ClassThresholds instance or float fallback.
    """
    try:
        from rfdetr import ClassThresholds
        if thresholds:
            return ClassThresholds(thresholds, default=default_threshold)
        return ClassThresholds({"person": 0.35}, default=default_threshold)
    except (ImportError, AttributeError):
        # Fallback if rfdetr.ClassThresholds is not available in mock/test environment
        class _FallbackClassThresholds:
            def __init__(self, mapping: Dict[str, float], default: float = 0.5):
                self.mapping = mapping
                self.default = default

            def get(self, class_name: str, default: Optional[float] = None) -> float:
                return self.mapping.get(class_name, self.default if default is None else default)

            def __repr__(self) -> str:
                return f"ClassThresholds({self.mapping}, default={self.default})"

        return _FallbackClassThresholds(thresholds or {"person": 0.35}, default=default_threshold)
