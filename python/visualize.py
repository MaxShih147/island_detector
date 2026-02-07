"""
Debug 3D visualizer for island detection C++ output.

Reads JSON exported by island_detect_cli and plots contours in 3D.

Usage:
    # Run C++ detector with JSON output:
    ./build/island_detect_cli data/layers 0.05 0.0 0.5 --json data/islands.json

    # Visualize:
    python visualize.py data/islands.json
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_islands(json_path: str):
    """Load island data from C++ CLI JSON output."""
    with open(json_path) as f:
        data = json.load(f)

    islands = []
    for item in data:
        contour = np.array(item["contour"], dtype=float)  # Nx2, world-space mm
        islands.append({
            "label": item["label"],
            "z": item["z"],
            "contour": contour,
        })

    return islands


def visualize_3d(islands):
    """3D plot of island contours at their world-space positions."""
    if not islands:
        print("No islands to show.")
        return

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors

    for isl in islands:
        contour = isl["contour"]
        z = isl["z"]
        label = isl["label"]
        color = colors[label % len(colors)]

        # Close the polygon
        pts = np.vstack([contour, contour[0:1]])
        zs = np.full(len(pts), z)

        # Draw contour outline
        ax.plot(pts[:, 0], pts[:, 1], zs, color=color, linewidth=1.5)

        # Draw filled polygon
        verts = [list(zip(pts[:, 0], pts[:, 1], zs))]
        poly = Poly3DCollection(verts, alpha=0.3, facecolor=color, edgecolor=color)
        ax.add_collection3d(poly)

        # Label
        cx, cy = contour.mean(axis=0)
        ax.text(cx, cy, z, f"#{label}", fontsize=7, color=color)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Island Contours — {len(islands)} islands (from C++ detector)")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize C++ island detection output")
    parser.add_argument("json_file", help="JSON file from island_detect_cli --json")
    args = parser.parse_args()

    islands = load_islands(args.json_file)
    print(f"Loaded {len(islands)} islands from {args.json_file}")
    visualize_3d(islands)


if __name__ == "__main__":
    main()
