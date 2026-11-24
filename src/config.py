"""Configuration schema and validation for WorkZone Vision.

Supported RF-DETR model architectures:
RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_MODELS = {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"}


class ModelConfig(BaseModel):
    """RF-DETR Model configuration."""
    name: str = Field(
        default="RFDETRLarge",
        description="Model architecture. Allowed: RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge",
    )
    pretrain_weights: Optional[str] = Field(
        default=None,
        description="Path to local pretrained checkpoint, or null for default",
    )
    device: str = Field(
        default="auto",
        description="Target device: 'auto' (CUDA if available else CPU), 'cuda', or 'cpu'",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model name '{v}'. Allowed Apache-2.0 architectures: {sorted(ALLOWED_MODELS)}. "
                f"Supported models are RFDETRNano, RFDETRSmall, RFDETRMedium, and RFDETRLarge."
            )
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Device must be 'auto', 'cuda', or 'cpu', got '{v}'")
        return v_lower


class PreprocessConfig(BaseModel):
    """Low-light and glare frame preprocessing configuration."""
    enable_clahe: bool = Field(default=True, description="Enable LAB-color space CLAHE enhancement")
    clip_limit: float = Field(default=2.0, ge=0.1, le=20.0, description="Contrast limit for CLAHE")
    tile_grid: List[int] = Field(default_factory=lambda: [8, 8], description="Grid size for histogram equalization")

    @field_validator("tile_grid")
    @classmethod
    def validate_tile_grid(cls, v: List[int]) -> List[int]:
        if len(v) != 2 or any(x < 1 for x in v):
            raise ValueError("tile_grid must contain exactly two positive integers [grid_x, grid_y]")
        return v


class TilingConfig(BaseModel):
    """Sliced / Tiled inference configuration for small/far-field targets."""
    enable_tiling: bool = Field(default=False, description="Enable sliced/tiled inference for far-field/small targets")
    slice_size: int = Field(default=640, ge=128, le=2048, description="Width and height of image tiles")
    overlap_ratio: float = Field(default=0.25, ge=0.0, lt=0.8, description="Fraction of tile overlap")
    iou_threshold: float = Field(default=0.55, ge=0.0, le=1.0, description="IoU threshold for box fusion")


class TTAConfig(BaseModel):
    """Test-Time Augmentation configuration."""
    enable_tta: bool = Field(default=False, description="Enable Test-Time Augmentation (horizontal flip)")
    flip_horizontal: bool = Field(default=True, description="Perform horizontal flip augmentation")


class DetectionConfig(BaseModel):
    """Detection inference configuration."""
    classes: List[str] = Field(
        default_factory=lambda: ["person"],
        description="Target classes for custom vision domains (e.g. ['person'], ['person', 'head'], or custom classes)",
    )
    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for person detection",
    )
    default_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Fallback threshold for non-specified classes",
    )
    class_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"person": 0.35, "head": 0.35},
        description="Per-class detection thresholds",
    )


class ExtractionConfig(BaseModel):
    """Video frame extraction and quality filtering configuration."""
    raw_video_dir: str = Field(
        default="data/raw_videos",
        description="Directory containing raw video clips or stream recordings",
    )
    extracted_frames_dir: str = Field(
        default="data/extracted_frames",
        description="Output directory for filtered candidate frames",
    )
    frame_interval: int = Field(
        default=15,
        ge=1,
        description="Sample every Nth video frame",
    )
    blur_threshold: float = Field(
        default=100.0,
        ge=0.0,
        description="Laplacian variance threshold. Lower values reject more blurry frames",
    )
    duplicate_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold above which adjacent frames are considered duplicates",
    )
    min_frame_change_pct: float = Field(
        default=1.5,
        ge=0.0,
        le=100.0,
        description="Minimum percentage of pixels that must change between sampled frames",
    )


class DatasetConfig(BaseModel):
    """Dataset path and validation configuration."""
    dataset_dir: str = Field(
        default="data/dataset",
        description="Root dataset directory containing train/val/test splits",
    )
    train_split: str = Field(default="train")
    val_split: str = Field(default="val")
    test_split: str = Field(default="test")
    annotation_file: str = Field(
        default="_annotations.coco.json",
        description="COCO annotation file name within split directories",
    )
    min_box_width: int = Field(default=8, ge=1)
    min_box_height: int = Field(default=16, ge=1)


class TrainingConfig(BaseModel):
    """Model fine-tuning configuration."""
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=2, ge=1)
    grad_accum_steps: int = Field(default=8, ge=1)
    lr: float = Field(default=0.00008, gt=0.0)
    early_stopping: int = Field(default=5, ge=1)
    output_dir: str = Field(default="models/checkpoints")
    resume_path: Optional[str] = Field(default=None)


class HardMiningConfig(BaseModel):
    """Hard-negative and low-confidence frame mining configuration."""
    input_frames_dir: str = Field(default="data/extracted_frames")
    output_dir: str = Field(default="data/hard_negatives")
    min_confidence: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Lower confidence threshold for uncertain detections",
    )
    max_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Upper confidence threshold for uncertain detections",
    )
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_samples_per_video: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def validate_confidence_range(self) -> "HardMiningConfig":
        if self.min_confidence >= self.max_confidence:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) must be less than "
                f"max_confidence ({self.max_confidence})"
            )
        return self


class EvaluationConfig(BaseModel):
    """Benchmarking and evaluation configuration."""
    test_images_dir: str = Field(default="data/dataset/test")
    test_annotation_file: str = Field(default="data/dataset/test/_annotations.coco.json")
    iou_thresholds: List[float] = Field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    benchmark_runs: int = Field(default=50, ge=1)
    warmup_runs: int = Field(default=10, ge=0)


class ExportConfig(BaseModel):
    """ONNX export and optimization configuration."""
    checkpoint_path: str = Field(default="models/checkpoints/best.pt")
    output_dir: str = Field(default="models/exported")
    format: str = Field(default="onnx")
    opset: int = Field(default=17, ge=11)
    simplify: bool = Field(default=True)

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower() != "onnx":
            raise ValueError(f"Export format must be 'onnx', got '{v}'")
        return v.lower()


class PipelineConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    tta: TTAConfig = Field(default_factory=TTAConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hard_mining: HardMiningConfig = Field(default_factory=HardMiningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    def resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' using torch."""
        if self.model.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.model.device


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from YAML file or return default.
    
    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml.
        
    Returns:
        Validated PipelineConfig object.
    """
    if config_path is None:
        default_path = Path("configs/config.yaml")
        if not default_path.exists():
            default_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        config_path = str(default_path)

    path = Path(config_path)
    if not path.is_file():
        # Fall back to default config if file does not exist yet
        return PipelineConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(**data)
