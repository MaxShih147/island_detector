"""
Debug visualization for island detection.

Loads layer PNGs, runs island detection (pure Python/OpenCV),
and visualizes results with matplotlib.

Usage:
    python visualize.py <png_dir> [--layer-height 0.05] [--stride 1]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def detect_islands_pair(img_below: np.ndarray, img_current: np.ndarray):
    """Detect islands between two binary layer images.

    Returns list of dicts: { contour: np.ndarray (Nx1x2), area: int }
    """
    # Binarize
    _, bin_below = cv2.threshold(img_below, 127, 255, cv2.THRESH_BINARY)
    _, bin_current = cv2.threshold(img_current, 127, 255, cv2.THRESH_BINARY)

    # Connected components on current layer
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_current, connectivity=8, ltype=cv2.CV_32S
    )

    islands = []

    for lbl in range(1, num_labels):
        left = stats[lbl, cv2.CC_STAT_LEFT]
        top = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        # Check overlap with layer below in bounding box region
        roi_labels = labels[top : top + h, left : left + w]
        roi_below = bin_below[top : top + h, left : left + w]
        component_mask = roi_labels == lbl
        overlap = np.any(component_mask & (roi_below > 0))

        if overlap:
            continue

        # Island found — extract contour
        mask = np.zeros_like(bin_current)
        mask[labels == lbl] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Pick largest contour
        best = max(contours, key=lambda c: len(c))

        islands.append({
            "contour": best.reshape(-1, 2),  # Nx2 array of (x, y) pixel coords
            "area": area,
            "centroid": (centroids[lbl, 0], centroids[lbl, 1]),
        })

    return islands


def run_detection(png_dir: str, stride: int = 1):
    """Load PNGs and detect islands across all layer pairs."""
    png_files = sorted(
        [f for f in Path(png_dir).glob("*.png")],
        key=lambda p: int(p.stem),
    )

    if len(png_files) < 2:
        print(f"Need at least 2 PNGs, found {len(png_files)}")
        sys.exit(1)

    # Apply stride
    if stride > 1:
        png_files = png_files[::stride]

    print(f"Processing {len(png_files)} layers (stride={stride})")

    # Load all images
    images = []
    for p in png_files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read: {p}")
            sys.exit(1)
        images.append(img)

    # Detect islands for each consecutive pair
    all_results = []  # list of (layer_idx, islands)
    for i in range(1, len(images)):
        islands = detect_islands_pair(images[i - 1], images[i])
        if islands:
            all_results.append((i, islands))

    return all_results, images, png_files


def visualize_results(all_results, images, png_files, layer_height: float):
    """Show layers with islands highlighted."""
    if not all_results:
        print("No islands detected.")
        return

    total = sum(len(islands) for _, islands in all_results)
    print(f"Total islands: {total} across {len(all_results)} layers")

    # Show up to 12 layers with islands
    n_show = min(len(all_results), 12)
    cols = min(n_show, 4)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(n_show):
        layer_idx, islands = all_results[idx]
        ax = axes[idx // cols, idx % cols]

        # Show current layer image
        img = images[layer_idx]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        # Draw island contours in red
        for isl in islands:
            contour = isl["contour"]
            # Close the polygon
            poly = np.vstack([contour, contour[0:1]])
            ax.plot(poly[:, 0], poly[:, 1], "r-", linewidth=1.5)
            cx, cy = isl["centroid"]
            ax.plot(cx, cy, "r+", markersize=8)

        z = layer_idx * layer_height
        ax.set_title(f"Layer {layer_idx} (z={z:.2f}mm)\n{len(islands)} island(s)")
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_show, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"Island Detection — {total} islands across {len(all_results)} layers")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Island detection debug visualizer")
    parser.add_argument("png_dir", help="Directory containing layer PNGs")
    parser.add_argument("--layer-height", type=float, default=0.05,
                        help="Layer height in mm (default: 0.05)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Layer stride: 1=every layer, 5=every 5th, etc.")
    args = parser.parse_args()

    all_results, images, png_files = run_detection(args.png_dir, args.stride)
    visualize_results(all_results, images, png_files, args.layer_height)


if __name__ == "__main__":
    main()
