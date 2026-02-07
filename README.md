# Island Detector

C++ library for detecting unsupported regions (islands) in SLA/DLP 3D printing layer images.

An **island** is a connected pixel region in layer N that has **zero overlap** with any pixel in layer N-1 (the layer below). These regions will fail to print without supports.

## Dependencies

- **OpenCV** (>= 4.5) — `core`, `imgproc`, `imgcodecs`
- **Clipper2** — fetched automatically via CMake FetchContent
- **CMake** (>= 3.16)

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

This produces:
- `libisland_detector.a` — static library
- `island_detect_cli` — CLI tool for testing

## Library API

### Header

```cpp
#include <island_detector.h>
```

### Data Structures

```cpp
namespace island {

struct Point2f { float x; float y; };

struct BBox3D {
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
};

struct Island {
  int label;                        // Global ID: 0..N-1, sorted by z
  std::vector<Point2f> contour;     // World-space contour (mm)
  float z;                          // World-space Z (mm)
};

struct DetectionConfig {
  float display_width;              // Physical display width (mm), e.g. 68.04
  float display_height;             // Physical display height (mm), e.g. 120.96
  float layer_height;               // Layer height (mm)
  BBox3D model_bbox;                // Model bounding box in world space
  float offset_mm = 0.0f;           // Contour offset/expansion in mm
};

}
```

### Function

```cpp
std::vector<island::Island> island::detect_islands(
    const std::vector<cv::Mat>& layer_images,
    const island::DetectionConfig& config);
```

### Integration Example

```cpp
#include <island_detector.h>
#include <opencv2/imgcodecs.hpp>

// 1. Load your sliced layer PNGs (grayscale, binary)
std::vector<cv::Mat> layers;
for (int i = 0; i < num_layers; ++i) {
    layers.push_back(cv::imread(layer_paths[i], cv::IMREAD_GRAYSCALE));
}

// 2. Configure
island::DetectionConfig config;
config.display_width  = 68.04f;   // Prusa SL1 display width (mm)
config.display_height = 120.96f;  // Prusa SL1 display height (mm)
config.layer_height   = 0.05f;    // mm per layer
config.model_bbox     = { min_x, min_y, min_z, max_x, max_y, max_z };
config.offset_mm      = 0.5f;     // expand contours 0.5mm outward

// 3. Detect
auto islands = island::detect_islands(layers, config);

// 4. Use results — each island has world-space contour + z
for (const auto& isl : islands) {
    // isl.label    — global ID (0..N-1)
    // isl.z        — world-space Z height (mm)
    // isl.contour  — vector<Point2f>, closed polygon in world-space XY (mm)
    //
    // For OpenGL: extrude each contour polygon at its Z height
    // to highlight the unsupported region on the 3D model.
    draw_island_overlay(isl.contour, isl.z);
}
```

### Input Details

| Parameter | Description |
|---|---|
| `layer_images` | Vector of `cv::Mat` (grayscale). Index 0 = bottom layer. Caller handles stride filtering — if you want to skip layers, filter the vector and multiply `layer_height` accordingly. |
| `display_width` | Physical display width in mm. Maps to PNG width (short axis). |
| `display_height` | Physical display height in mm. Maps to PNG height (long axis). |
| `layer_height` | Layer height in mm. If you skip every N layers, pass `N * original_layer_height`. |
| `model_bbox` | Model bounding box after all transforms (rotation, translation, scale). Used for: `min_z` as base Z, center XY for coordinate offset. |
| `offset_mm` | Clipper2 polygon expansion in mm. Islands are often tiny — offset makes them visible. 0 = raw contour. |

### Output Details

| Field | Description |
|---|---|
| `label` | Global sequential ID: 0 to N-1, sorted by Z ascending. |
| `contour` | Closed polygon in world-space mm. Ready for OpenGL rendering. |
| `z` | World-space Z position (mm) = `model_bbox.min_z + layer_index * layer_height`. |

### Coordinate Mapping

The library converts pixel contours to world-space using the rotated display mapping:

```
scene_x = display_h/2 - (py / img_h) * display_h + center_x
scene_y = (px / img_w) * display_w - display_w/2 + center_y
z       = model_bbox.min_z + layer_index * layer_height
```

Where `center_x`, `center_y` are computed from `model_bbox` center.

This matches the SLA printer's rotated display orientation (PNG width = short physical axis, PNG height = long physical axis).

### CMake Integration

```cmake
add_subdirectory(island_detector)
target_link_libraries(your_app PRIVATE island_detector)
```

## CLI Tool

For testing and debug JSON export:

```bash
# Basic detection
./build/island_detect_cli <png_dir> <layer_height> <min_z> [offset_mm]

# With JSON export for Python visualizer
./build/island_detect_cli data/layers 0.05 0.0 0.5 --json data/islands.json
```

## Python Debug Visualizer

Reads JSON from the C++ CLI and plots contours in 3D (matplotlib):

```bash
# 1. Run C++ detector
./build/island_detect_cli data/layers 0.05 0.0 0.5 --json data/islands.json

# 2. Visualize
pip install matplotlib numpy
python python/visualize.py data/islands.json
```

The 3D plot shows each island contour at its Z height, colored by label. You can rotate the view to inspect positions.
