"""Low-light and glare frame preprocessing for WorkZone Vision.

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
Luminance channel in LAB color space to recover detail in shadowed corners,
backlit scenes, and glaring light sources without altering chromatic balance.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import PreprocessConfig, load_config


def enhance_cctv_frame(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance illumination balance and contrast in an RGB image.
    
    Operates strictly on the L-channel in LAB color space to preserve
    hue and chromatic fidelity while boosting dynamic range.
    
    Args:
        image_rgb: Input RGB image as uint8 numpy ndarray (H, W, 3).
        clip_limit: Threshold for contrast limiting in CLAHE. Higher values
                    yield stronger local contrast.
        tile_grid: Grid size for histogram equalization (e.g. (8, 8)).
        
    Returns:
        Enhanced RGB image as uint8 numpy ndarray (H, W, 3).
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"Expected image_rgb to be numpy.ndarray, got {type(image_rgb)}")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {image_rgb.shape}")

    # Ensure contiguous uint8 array
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Initialize and apply CLAHE to the Luminance channel only
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
    )
    l_enhanced = clahe.apply(l_channel)

    # Merge channels back and convert to RGB
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return enhanced_rgb


# Alias for general video streams
enhance_frame = enhance_cctv_frame


def preprocess_image(
    image_rgb: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Apply configured preprocessing pipeline on an RGB frame.
    
    Args:
        image_rgb: Input RGB image.
        config: PreprocessConfig settings. Defaults to loaded config.
        
    Returns:
        Processed RGB image ready for detector inference.
    """
    if config is None:
        cfg = load_config()
        config = cfg.preprocess

    if not config.enable_clahe:
        return image_rgb

    grid = (config.tile_grid[0], config.tile_grid[1])
    return enhance_cctv_frame(
        image_rgb=image_rgb,
        clip_limit=config.clip_limit,
        tile_grid=grid,
    )
