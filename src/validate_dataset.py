"""COCO Dataset integrity and quality validation for RF-DETR fine-tuning.

Verifies:
1. Split existence (train, val/valid, optional test).
2. COCO JSON syntax and required fields.
3. Image file existence on disk, readable headers, and dimension agreement.
4. Bounding box validity (x, y, w, h within bounds, non-zero area).
5. Target category mapping ('person', optional 'head').
6. Class distribution and box size statistics (small, medium, large).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DatasetConfig, load_config

console = Console()


def find_annotation_file(split_dir: Path, preferred_name: str = "_annotations.coco.json") -> Optional[Path]:
    """Find annotation file inside a split directory or subfolder.
    
    Checks:
    - split_dir / preferred_name
    - split_dir / "annotations.json"
    - split_dir / "instances_train.json" or similar
    - split_dir / "annotations" / "*.json"
    """
    candidate = split_dir / preferred_name
    if candidate.is_file():
        return candidate

    direct_jsons = list(split_dir.glob("*.json"))
    if direct_jsons:
        return direct_jsons[0]

    sub_ann = split_dir / "annotations"
    if sub_ann.is_dir():
        ann_jsons = list(sub_ann.glob("*.json"))
        if ann_jsons:
            return ann_jsons[0]

    return None


def validate_split(
    split_name: str,
    split_dir: Path,
    annotation_path: Path,
    target_classes: List[str],
    min_box_w: int = 8,
    min_box_h: int = 16,
) -> Dict[str, Any]:
    """Validate a single dataset split.
    
    Args:
        split_name: Name of split ('train', 'val', 'test').
        split_dir: Directory where images reside.
        annotation_path: Path to COCO JSON file.
        target_classes: Allowed class names (e.g. ['person', 'head']).
        min_box_w: Minimum valid box width.
        min_box_h: Minimum valid box height.
        
    Returns:
        Validation statistics and error logs for the split.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not annotation_path.is_file():
        errors.append(f"Annotation file not found: {annotation_path}")
        return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse JSON file {annotation_path}: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

    for required_key in ("images", "annotations", "categories"):
        if required_key not in coco_data:
            errors.append(f"Missing required top-level COCO key '{required_key}' in {annotation_path.name}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

    categories = coco_data.get("categories", [])
    cat_id_to_name: Dict[int, str] = {}
    for cat in categories:
        cid = cat.get("id")
        cname = cat.get("name")
        if cid is not None and cname is not None:
            cat_id_to_name[cid] = cname

    target_class_set = set(target_classes)
    dataset_class_set = set(cat_id_to_name.values())

    missing_classes = target_class_set - dataset_class_set
    if missing_classes:
        warnings.append(f"Split '{split_name}' missing expected target classes: {missing_classes}")

    extra_classes = dataset_class_set - target_class_set
    if extra_classes:
        warnings.append(f"Split '{split_name}' contains extra classes not in target list: {extra_classes}")

    images = coco_data.get("images", [])
    image_id_to_meta: Dict[Any, Dict[str, Any]] = {}
    corrupted_images = 0
    missing_image_files = 0
    dim_mismatches = 0

    for img in images:
        iid = img.get("id")
        fname = img.get("file_name")
        w = img.get("width")
        h = img.get("height")

        if iid is None or fname is None:
            errors.append(f"Image entry missing id or file_name: {img}")
            continue

        image_id_to_meta[iid] = img
        img_path = split_dir / fname
        if not img_path.is_file():
            # Also check if image is located in parent or sibling
            alt_path = annotation_path.parent / fname
            if alt_path.is_file():
                img_path = alt_path
            else:
                missing_image_files += 1
                if missing_image_files <= 5:
                    errors.append(f"Image file not found on disk: {img_path}")
                continue

        # Verify image readability and dimensions
        try:
            with Image.open(img_path) as pil_img:
                actual_w, actual_h = pil_img.size
                if w is not None and h is not None:
                    if actual_w != w or actual_h != h:
                        dim_mismatches += 1
                        if dim_mismatches <= 5:
                            warnings.append(
                                f"Dimension mismatch in {fname}: JSON has {w}x{h}, disk has {actual_w}x{actual_h}"
                            )
        except Exception as e:
            corrupted_images += 1
            if corrupted_images <= 5:
                errors.append(f"Corrupted or unreadable image {img_path}: {e}")

    annotations = coco_data.get("annotations", [])
    valid_boxes = 0
    invalid_boxes = 0
    degenerate_boxes = 0
    out_of_bounds_boxes = 0
    class_counts: Dict[str, int] = {cname: 0 for cname in cat_id_to_name.values()}

    # Area distribution
    small_boxes = 0   # area < 32^2 (1024)
    medium_boxes = 0  # 32^2 <= area < 96^2 (9216)
    large_boxes = 0   # area >= 96^2

    annotated_image_ids: Set[Any] = set()

    for ann in annotations:
        aid = ann.get("id")
        iid = ann.get("image_id")
        cid = ann.get("category_id")
        bbox = ann.get("bbox")

        if iid not in image_id_to_meta:
            errors.append(f"Annotation {aid} references non-existent image_id: {iid}")
            invalid_boxes += 1
            continue

        if cid not in cat_id_to_name:
            errors.append(f"Annotation {aid} references non-existent category_id: {cid}")
            invalid_boxes += 1
            continue

        if not bbox or len(bbox) != 4:
            errors.append(f"Annotation {aid} has invalid bbox format: {bbox}")
            invalid_boxes += 1
            continue

        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            degenerate_boxes += 1
            invalid_boxes += 1
            continue

        if bw < min_box_w or bh < min_box_h:
            degenerate_boxes += 1

        img_meta = image_id_to_meta[iid]
        img_w = img_meta.get("width", 1920)
        img_h = img_meta.get("height", 1080)

        # Check bounds
        if x < 0 or y < 0 or (x + bw) > (img_w + 5) or (y + bh) > (img_h + 5):
            out_of_bounds_boxes += 1

        cname = cat_id_to_name[cid]
        class_counts[cname] = class_counts.get(cname, 0) + 1
        annotated_image_ids.add(iid)
        valid_boxes += 1

        area = bw * bh
        if area < 1024:
            small_boxes += 1
        elif area < 9216:
            medium_boxes += 1
        else:
            large_boxes += 1

    empty_images = len(image_id_to_meta) - len(annotated_image_ids)
    if empty_images > 0:
        warnings.append(
            f"Split '{split_name}' contains {empty_images} background images without annotations "
            f"(valid negatives for training)."
        )

    if missing_image_files > 5:
        errors.append(f"Total missing image files on disk in '{split_name}': {missing_image_files}")
    if corrupted_images > 5:
        errors.append(f"Total corrupted image files in '{split_name}': {corrupted_images}")

    is_valid = len(errors) == 0

    stats = {
        "split": split_name,
        "total_images": len(images),
        "total_annotations": len(annotations),
        "valid_boxes": valid_boxes,
        "invalid_boxes": invalid_boxes,
        "degenerate_boxes": degenerate_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "empty_images": empty_images,
        "class_counts": class_counts,
        "box_sizes": {
            "small (<32px)": small_boxes,
            "medium (32-96px)": medium_boxes,
            "large (>96px)": large_boxes,
        },
    }

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def validate_cctv_dataset(
    dataset_dir: Optional[str] = None,
    config: Optional[DatasetConfig] = None,
    target_classes: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate full COCO dataset directory across train, val, and test splits.
    
    Args:
        dataset_dir: Root dataset folder.
        config: DatasetConfig instance.
        target_classes: Target classes (defaults to ['person']).
        
    Returns:
        Tuple of (is_valid_boolean, detailed_report_dict).
    """
    if config is None:
        cfg = load_config()
        config = cfg.dataset

    if target_classes is None:
        cfg = load_config()
        target_classes = cfg.detection.classes

    d_path = Path(dataset_dir or config.dataset_dir)
    console.print(
        Panel(
            f"[bold cyan]Validating Dataset Integrity (COCO Format)[/bold cyan]\n"
            f"Directory: [yellow]{d_path.resolve()}[/yellow]\n"
            f"Target Classes: [green]{target_classes}[/green]",
            title="Dataset Validator",
            expand=False,
        )
    )

    if not d_path.is_dir():
        console.print(f"[bold red]Dataset directory does not exist:[/bold red] {d_path}")
        return False, {"overall_valid": False, "error": f"Directory not found: {d_path}"}

    splits_to_check = [
        ("train", config.train_split, True),
        ("val", config.val_split, True),
        ("test", config.test_split, False),
    ]

    all_split_reports: Dict[str, Any] = {}
    overall_valid = True

    # Summary table
    table = Table(title="Dataset Split Validation Overview")
    table.add_column("Split", style="cyan")
    table.add_column("Images", justify="right")
    table.add_column("Annotations", justify="right")
    table.add_column("Classes", style="magenta")
    table.add_column("Empty Imgs", justify="right")
    table.add_column("Status", justify="center")

    for key, folder_name, is_required in splits_to_check:
        split_dir = d_path / folder_name
        # Also check 'valid' if 'val' does not exist
        if not split_dir.is_dir() and folder_name == "val":
            alt_val = d_path / "valid"
            if alt_val.is_dir():
                split_dir = alt_val

        if not split_dir.is_dir():
            if is_required:
                overall_valid = False
                console.print(f"[bold red]Required split '{folder_name}' not found in {d_path}[/bold red]")
                all_split_reports[key] = {
                    "valid": False,
                    "errors": [f"Required split folder '{folder_name}' not found"],
                    "warnings": [],
                    "stats": {},
                }
            continue

        ann_file = find_annotation_file(split_dir, preferred_name=config.annotation_file)
        if not ann_file:
            if is_required:
                overall_valid = False
            all_split_reports[key] = {
                "valid": False,
                "errors": [f"No annotation JSON found in {split_dir}"],
                "warnings": [],
                "stats": {},
            }
            continue

        res = validate_split(
            split_name=key,
            split_dir=split_dir,
            annotation_path=ann_file,
            target_classes=target_classes,
            min_box_w=config.min_box_width,
            min_box_h=config.min_box_height,
        )

        all_split_reports[key] = res
        if not res["valid"]:
            overall_valid = False

        stats = res.get("stats", {})
        c_str = ", ".join(f"{k}:{v}" for k, v in stats.get("class_counts", {}).items()) or "None"
        status_str = "[bold green]PASS[/bold green]" if res["valid"] else "[bold red]FAIL[/bold red]"

        table.add_row(
            key,
            str(stats.get("total_images", 0)),
            str(stats.get("total_annotations", 0)),
            c_str,
            str(stats.get("empty_images", 0)),
            status_str,
        )

    console.print(table)

    # Print detailed errors / warnings
    for key, rep in all_split_reports.items():
        if rep.get("errors"):
            console.print(f"[bold red]Errors in split '{key}':[/bold red]")
            for err in rep["errors"]:
                console.print(f"  • {err}")
        if rep.get("warnings"):
            console.print(f"[yellow]Warnings in split '{key}':[/yellow]")
            for warn in rep["warnings"]:
                console.print(f"  • {warn}")

    report = {
        "overall_valid": overall_valid,
        "dataset_dir": str(d_path.resolve()),
        "target_classes": target_classes,
        "splits": all_split_reports,
    }

    # Save report JSON
    rep_path = d_path / "dataset_validation_report.json"
    try:
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"[dim]Validation report written to: {rep_path}[/dim]")
    except Exception:
        pass

    return overall_valid, report


# General alias
validate_dataset = validate_cctv_dataset


if __name__ == "__main__":
    validate_cctv_dataset()
