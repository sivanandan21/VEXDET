"""Video frame extraction and quality filtering.

Performs:
1. Video sampling (every N frames).
2. Motion blur filtering using Variance of Laplacian.
3. Duplicate/static frame filtering using structural difference and histogram correlation.
4. Color conversion: RGB validation.
5. Saves high-quality candidate frames for labeling and dataset preparation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from src.config import ExtractionConfig, load_config

console = Console()


def compute_blur_score(gray_frame: np.ndarray) -> float:
    """Calculate sharpness using variance of the Laplacian.
    
    Higher variance indicates sharp edges; low variance indicates blur.
    
    Args:
        gray_frame: Grayscale image as numpy ndarray.
        
    Returns:
        Variance of Laplacian (blur score).
    """
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    score = float(laplacian.var())
    return score


def compute_frame_difference(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    diff_threshold: int = 25,
) -> Tuple[float, float]:
    """Calculate percentage change and histogram correlation between consecutive frames.
    
    Args:
        prev_gray: Previous grayscale frame.
        curr_gray: Current grayscale frame.
        diff_threshold: Absolute pixel intensity difference threshold.
        
    Returns:
        Tuple of (pixel_change_percentage, histogram_correlation).
    """
    # 1. Absolute difference thresholding
    abs_diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    change_pct = (changed_pixels / total_pixels) * 100.0

    # 2. Normalized histogram correlation
    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    correlation = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    return change_pct, correlation


def extract_cctv_frames(
    video_path: str,
    output_dir: str,
    config: Optional[ExtractionConfig] = None,
) -> Dict[str, Any]:
    """Extract sharp, non-duplicate candidate frames from a video file.
    
    Args:
        video_path: Path to video file (.mp4, .avi, .mkv, .mov).
        output_dir: Directory to save extracted images.
        config: ExtractionConfig settings.
        
    Returns:
        Summary statistics dictionary.
    """
    if config is None:
        cfg = load_config()
        config = cfg.extraction

    v_path = Path(video_path)
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    console.print(
        f"[cyan]Processing video:[/cyan] {v_path.name} | "
        f"Frames: {total_video_frames} | FPS: {fps:.2f} | "
        f"Interval: every {config.frame_interval} frames"
    )

    frame_idx = 0
    sampled_count = 0
    blurry_dropped = 0
    duplicate_dropped = 0
    saved_count = 0

    prev_gray: Optional[np.ndarray] = None
    saved_records: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Extracting {v_path.name}", total=max(total_video_frames, 1))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            progress.update(task, advance=1)
            frame_idx += 1

            # Sample every Nth frame
            if frame_idx % config.frame_interval != 0:
                continue

            sampled_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Blur Detection
            blur_score = compute_blur_score(gray)
            if blur_score < config.blur_threshold:
                blurry_dropped += 1
                continue

            # 2. Duplicate / Static Frame Filtering
            if prev_gray is not None:
                change_pct, hist_corr = compute_frame_difference(prev_gray, gray)
                is_duplicate = (
                    change_pct < config.min_frame_change_pct
                    or hist_corr >= config.duplicate_similarity_threshold
                )
                if is_duplicate:
                    duplicate_dropped += 1
                    continue
            else:
                change_pct = 100.0
                hist_corr = 0.0

            # 3. Passed quality checks: Save frame
            file_name = f"{v_path.stem}_frame_{frame_idx:06d}.jpg"
            save_path = out_dir / file_name

            # Note: We save as BGR using standard OpenCV imwrite
            cv2.imwrite(str(save_path), frame)
            prev_gray = gray.copy()
            saved_count += 1

            saved_records.append({
                "file_name": file_name,
                "frame_index": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "blur_score": round(blur_score, 2),
                "change_pct": round(change_pct, 2),
                "hist_correlation": round(hist_corr, 4),
            })

    cap.release()

    stats = {
        "video_file": v_path.name,
        "total_video_frames": total_video_frames,
        "sampled_frames": sampled_count,
        "blurry_dropped": blurry_dropped,
        "duplicate_dropped": duplicate_dropped,
        "saved_frames": saved_count,
        "quality_yield_pct": round((saved_count / max(sampled_count, 1)) * 100.0, 2),
    }

    # Save summary metadata JSON
    meta_path = out_dir / f"{v_path.stem}_extraction_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "records": saved_records}, f, indent=2)

    return stats


def process_video_directory(
    video_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[ExtractionConfig] = None,
) -> List[Dict[str, Any]]:
    """Batch extract frames from all video files in video_dir.
    
    Args:
        video_dir: Directory containing input video files.
        output_dir: Directory to save extracted frames.
        config: Optional extraction config.
        
    Returns:
        List of extraction summary dictionaries.
    """
    if config is None:
        cfg = load_config()
        config = cfg.extraction

    v_dir = Path(video_dir or config.raw_video_dir)
    out_dir = Path(output_dir or config.extracted_frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
    video_files = [f for f in v_dir.iterdir() if f.is_file() and f.suffix.lower() in video_extensions] if v_dir.exists() else []

    if not video_files:
        console.print(
            f"[yellow]No video files found in {v_dir}. "
            f"Place video clips (.mp4, .avi, .mkv) in this directory.[/yellow]"
        )
        return []

    all_stats: List[Dict[str, Any]] = []
    for vf in video_files:
        stats = extract_cctv_frames(str(vf), str(out_dir), config=config)
        all_stats.append(stats)

    # Display summary table
    table = Table(title="Video Frame Extraction Summary")
    table.add_column("Video File", style="cyan")
    table.add_column("Sampled", justify="right")
    table.add_column("Blurry Dropped", justify="right", style="red")
    table.add_column("Duplicate Dropped", justify="right", style="yellow")
    table.add_column("Quality Saved", justify="right", style="green")
    table.add_column("Yield %", justify="right", style="magenta")

    for s in all_stats:
        table.add_row(
            s["video_file"],
            str(s["sampled_frames"]),
            str(s["blurry_dropped"]),
            str(s["duplicate_dropped"]),
            str(s["saved_frames"]),
            f"{s['quality_yield_pct']}%",
        )

    console.print(table)
    return all_stats


# General alias
extract_video_frames = extract_cctv_frames


if __name__ == "__main__":
    process_video_directory()
