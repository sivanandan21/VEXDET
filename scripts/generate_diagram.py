"""Generate a high-resolution dark-themed pipeline architecture diagram."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output directory
docs_dir = Path(r"d:\project\live camera\workzone-vision\docs")
docs_dir.mkdir(parents=True, exist_ok=True)
output_path = docs_dir / "pipeline_architecture.png"

# Setup high-res figure with GitHub dark theme aesthetic
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200)
bg_color = "#0d1117"  # GitHub dark background
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Helper function to draw rounded stylized card
def draw_card(ax, x, y, w, h, title, subtitle, bullets, badge, border_color="#30363d", card_bg="#161b22", title_color="#58a6ff"):
    # Shadow / glow
    shadow = patches.FancyBboxPatch(
        (x + 0.05, y - 0.05), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.25",
        facecolor="#000000", edgecolor="none", alpha=0.35, zorder=1
    )
    ax.add_patch(shadow)

    # Main Card
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.25",
        facecolor=card_bg, edgecolor=border_color, linewidth=1.8, zorder=2
    )
    ax.add_patch(card)

    # Title & Badge
    ax.text(x + 0.2, y + h - 0.35, title, fontsize=12.5, fontweight="bold", color=title_color, zorder=3)
    if badge:
        # Badge pill
        bw = len(badge) * 0.12 + 0.4
        badge_box = patches.FancyBboxPatch(
            (x + w - bw - 0.2, y + h - 0.48), bw, 0.32,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#21262d", edgecolor="#30363d", linewidth=1, zorder=3
        )
        ax.add_patch(badge_box)
        ax.text(x + w - bw / 2 - 0.2, y + h - 0.34, badge, fontsize=8.5, fontweight="bold", color="#8b949e", ha="center", va="center", zorder=4)

    # Subtitle
    if subtitle:
        ax.text(x + 0.2, y + h - 0.68, subtitle, fontsize=9.5, fontweight="semibold", color="#c9d1d9", zorder=3)

    # Bullets
    line_y = y + h - 1.05
    for b in bullets:
        ax.text(x + 0.25, line_y, f"• {b}", fontsize=8.5, color="#8b949e", zorder=3)
        line_y -= 0.34

# Header Title
ax.text(0.5, 8.0, "WorkZone Vision — End-to-End Pipeline Architecture", fontsize=18, fontweight="bold", color="#f0f6fc", zorder=3)
ax.text(0.5, 7.6, "Edge-Ready RF-DETR Fine-Tuning, Active Learning & ONNX Optimization", fontsize=11, color="#8b949e", zorder=3)

# Top Row: Stage 1, 2, 3, 4
# 1. Video Feeds
draw_card(
    ax, x=0.5, y=4.5, w=3.1, h=2.5,
    title="1. Video Input", subtitle="Camera Feeds & Streams",
    bullets=[
        "RTSP Live Streams",
        "Drones & Action Cams",
        "Robotics & Edge Feeds",
        "Raw Video Files (.mp4, .avi)"
    ],
    badge="INGESTION", border_color="#1f6feb", title_color="#58a6ff"
)

# 2. Quality Extraction
draw_card(
    ax, x=4.1, y=4.5, w=3.2, h=2.5,
    title="2. Video Curation", subtitle="Sharpness & De-duplication",
    bullets=[
        "Laplacian Blur Filter (>100)",
        "Histogram Correlation (<0.95)",
        "Frame Delta Thresholding",
        "High-Yield Clean Candidates"
    ],
    badge="FILTERING", border_color="#238636", title_color="#3fb950"
)

# 3. Dataset Validation
draw_card(
    ax, x=7.8, y=4.5, w=3.2, h=2.5,
    title="3. COCO Auditing", subtitle="Integrity & Coordinates",
    bullets=[
        "Schema & Key Validation",
        "Degenerate Box Pruning",
        "Dimension & Bounds Check",
        "Class Balance Analytics"
    ],
    badge="VALIDATION", border_color="#8957e5", title_color="#a371f7"
)

# 4. Fine-Tuning
draw_card(
    ax, x=11.4, y=4.5, w=3.1, h=2.5,
    title="4. RF-DETR Training", subtitle="Detection Transformer",
    bullets=[
        "Apache-2.0 RT-DETR Base",
        "NMS-Free End-to-End Loss",
        "CUDA / CPU Auto Fallback",
        "Multi-Class Custom Tuning"
    ],
    badge="TRAINING", border_color="#d29922", title_color="#e3b341"
)

# Bottom Row: Active Learning, Benchmarking, Export
# 5. Active Learning & Hard-Negative Mining
draw_card(
    ax, x=11.4, y=1.2, w=3.1, h=2.5,
    title="5. Hard Mining", subtitle="Active Learning Loop",
    bullets=[
        "Ambiguity Band [0.15, 0.45]",
        "False-Positive Discovery",
        "Visual Overlays & Manifest",
        "Retraining Round 2 Prep"
    ],
    badge="ACTIVE LEARNING", border_color="#f85149", title_color="#ff7b72"
)

# 6. Benchmarking
draw_card(
    ax, x=6.0, y=1.2, w=4.5, h=2.5,
    title="6. Rigorous Benchmarking", subtitle="Precision, Recall & Latency",
    bullets=[
        "True mAP@50 & mAP@50:95 (101-point COCO interpolation)",
        "Statistical Latency: Min, Max, Mean, P95 (ms)",
        "Throughput Profiling (Frames Per Second)",
        "Side-by-Side Comparison vs Pretrained Weights"
    ],
    badge="EVALUATION", border_color="#da3633", title_color="#f85149"
)

# 7. ONNX Edge Export
draw_card(
    ax, x=0.5, y=1.2, w=4.8, h=2.5,
    title="7. ONNX Export & Edge Runtime", subtitle="Production Deployment",
    bullets=[
        "Export PyTorch Checkpoint to ONNX (opset 17)",
        "RFDETRInference Runtime Verification & Speedup",
        "Cross-Platform Deployment (Jetson, TensorRT, x86/ARM)",
        "Zero-Copy OpenCV BGR to RGB Streaming Pipeline"
    ],
    badge="DEPLOYMENT", border_color="#388bfd", title_color="#79c0ff"
)

# Connection Arrows Helper
def draw_arrow(ax, start, end, label=None, color="#388bfd", style="-|>", rad=0.0, lw=2.0, dashed=False):
    ls = "--" if dashed else "-"
    arrow = patches.FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"{style},head_length=6,head_width=4",
        color=color, linewidth=lw, linestyle=ls, zorder=5
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + (0.18 if rad == 0 else rad * 1.5)
        ax.text(mid_x, mid_y, label, fontsize=8.5, fontweight="bold", color=color, ha="center", va="center", zorder=6)

# Horizontal arrows top row
draw_arrow(ax, (3.6, 5.75), (4.1, 5.75), color="#58a6ff")
draw_arrow(ax, (7.3, 5.75), (7.8, 5.75), color="#3fb950")
draw_arrow(ax, (11.0, 5.75), (11.4, 5.75), color="#a371f7")

# Vertical arrow from Training (4) to Hard Mining (5)
draw_arrow(ax, (12.95, 4.5), (12.95, 3.7), label="Checkpoint", color="#e3b341")

# Feedback loop arrow from Hard Mining (5) back up to Training (4)
draw_arrow(ax, (14.2, 3.7), (14.2, 4.5), label="Round 2 Loop", color="#ff7b72", dashed=True)

# Horizontal arrow from Training/Hard Mining to Benchmarking (6)
draw_arrow(ax, (11.4, 2.45), (10.5, 2.45), label="Audit", color="#f85149")

# Horizontal arrow from Benchmarking (6) to Export (7)
draw_arrow(ax, (6.0, 2.45), (5.3, 2.45), label="Verified", color="#79c0ff")

plt.tight_layout()
plt.savefig(str(output_path), dpi=200, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
plt.close()
print(f"Diagram successfully generated at: {output_path}")
