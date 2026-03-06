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

#### `layer_images`

Vector of `cv::Mat` (grayscale, `CV_8UC1`). Index 0 = bottom layer (closest to build plate). The library compares consecutive pairs: layer `i` vs layer `i-1`, so layer 0 is always considered supported (on the build plate) and never produces islands.

If you want to skip layers for performance (e.g. check every 5th layer), filter the vector before passing it and multiply `layer_height` accordingly.

#### `DetectionConfig` Fields

| Field | Type | Example | Description |
|---|---|---|---|
| `display_width` | `float` | `68.04` | LCD screen physical width in mm |
| `display_height` | `float` | `120.96` | LCD screen physical height in mm |
| `layer_height` | `float` | `0.05` | Layer thickness in mm |
| `model_bbox` | `BBox3D` | `{-10,-15,0, 10,15,30}` | Model bounding box in world space (mm) |
| `offset_mm` | `float` | `0.5` | Contour outward expansion in mm (0 = disabled) |

**`display_width` / `display_height` — LCD screen physical dimensions**

SLA/DLP printers use an LCD screen to cure resin layer by layer. Each sliced layer PNG covers the full screen area. These values define the physical size of that screen in millimeters, which is needed to convert pixel positions in the PNG to real-world millimeter coordinates.

```
PNG image (pixels)              LCD screen (mm)
+------------------+            +------------------+
|                  |  -------> |                  |
|   1440 x 2560   |   mapped   |  68.04 x 120.96  |
|     (pixels)     |            |      (mm)        |
+------------------+            +------------------+

pixel_mm_x = display_width  / image_width_px   (mm per pixel, horizontal)
pixel_mm_y = display_height / image_height_px   (mm per pixel, vertical)
```

Common values by printer:
| Printer | `display_width` | `display_height` | Resolution |
|---|---|---|---|
| Prusa SL1/SL1S | 68.04 | 120.96 | 1440 x 2560 |
| Elegoo Mars 2 | 68.04 | 120.96 | 1440 x 2560 |
| Elegoo Saturn | 128.0 | 80.0 | 3840 x 2400 |

**`layer_height` — layer thickness**

The Z coordinate of each island is computed as:

```
z = model_bbox.min_z + layer_index * layer_height
```

If you pre-filter layers (e.g. check every 3rd layer for speed), pass `3 * original_layer_height` so the Z values remain correct.

**`model_bbox` — model bounding box in world space**

The bounding box of the 3D model after all transforms (rotation, translation, scale) have been applied. Used for two purposes:

1. **`min_z`**: The Z height of the model's bottom face. This is the base Z for computing each island's world-space Z coordinate.

2. **Center XY** (`(min_x+max_x)/2`, `(min_y+max_y)/2`): The model's center in the XY plane. The coordinate mapping adds this offset so that island contours are positioned correctly relative to the model in world space.

```
model_bbox:
  min = (-10.5, -15.2, 0.0)    center_x = 0.0
  max = ( 10.5,  15.2, 30.0)   center_y = 0.0    base_z = 0.0
```

If your model is centered at the origin, set `min_x = min_y = 0`, `max_x = max_y = 0` (center will be 0,0). The CLI tool uses this simplified form.

**`offset_mm` — contour outward expansion**

After converting island contours to world-space mm, optionally expand them outward using Clipper2's `InflatePaths` with round joins. This is useful because:

- Raw island contours can be very small (a few pixels wide)
- Support structures typically need to cover slightly beyond the island boundary
- A value of 0.3-0.5 mm is a reasonable default for support placement

Set to `0.0` to get the exact pixel-level contour with no expansion.

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
