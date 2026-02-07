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


def contour_to_world(contour_px, img_w, img_h, display_w, display_h):
    """Convert pixel contour (Nx2) to physical mm coordinates (Nx2).

    Maps pixel space to display physical space centered at origin:
      world_x = (px / img_w) * display_w - display_w / 2
      world_y = display_h / 2 - (py / img_h) * display_h   (Y inverted)
    """
    px = contour_px[:, 0].astype(float)
    py = contour_px[:, 1].astype(float)
    world_x = (px / img_w) * display_w - display_w / 2
    world_y = display_h / 2 - (py / img_h) * display_h
    return np.column_stack([world_x, world_y])


def visualize_3d(all_results, images, layer_height: float,
                 display_w: float = 68.04, display_h: float = 120.96):
    """3D plot of island contours at their world-space Z positions."""
    if not all_results:
        print("No islands to show in 3D.")
        return

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    img_h, img_w = images[0].shape[:2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors
    global_idx = 0

    for layer_idx, islands in all_results:
        z = layer_idx * layer_height

        for isl in islands:
            contour_px = isl["contour"]
            world_xy = contour_to_world(contour_px, img_w, img_h, display_w, display_h)

            # Close the polygon
            pts = np.vstack([world_xy, world_xy[0:1]])
            zs = np.full(len(pts), z)

            color = colors[global_idx % len(colors)]

            # Draw contour outline
            ax.plot(pts[:, 0], pts[:, 1], zs, color=color, linewidth=1.5)

            # Draw filled polygon
            verts = [list(zip(pts[:, 0], pts[:, 1], zs))]
            poly = Poly3DCollection(verts, alpha=0.3, facecolor=color, edgecolor=color)
            ax.add_collection3d(poly)

            # Label
            cx, cy = world_xy.mean(axis=0)
            ax.text(cx, cy, z, f"#{global_idx}", fontsize=7, color=color)

            global_idx += 1

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Island Contours in 3D — {global_idx} islands")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Island detection debug visualizer")
    parser.add_argument("png_dir", help="Directory containing layer PNGs")
    parser.add_argument("--layer-height", type=float, default=0.05,
                        help="Layer height in mm (default: 0.05)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Layer stride: 1=every layer, 5=every 5th, etc.")
    parser.add_argument("--display-w", type=float, default=68.04,
                        help="Display width in mm (default: 68.04)")
    parser.add_argument("--display-h", type=float, default=120.96,
                        help="Display height in mm (default: 120.96)")
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both",
                        help="Visualization mode (default: both)")
    args = parser.parse_args()

    all_results, images, png_files = run_detection(args.png_dir, args.stride)

    if args.mode in ("2d", "both"):
        visualize_results(all_results, images, png_files, args.layer_height)
    if args.mode in ("3d", "both"):
        visualize_3d(all_results, images, args.layer_height, args.display_w, args.display_h)


if __name__ == "__main__":
    main()
