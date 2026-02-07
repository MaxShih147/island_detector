#include "island_detector.h"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

static void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " <png_dir> <layer_height_mm> <min_z> [offset_mm]"
            << std::endl;
  std::cerr << "  png_dir        : directory containing layer PNGs" << std::endl;
  std::cerr << "  layer_height_mm: layer height in mm" << std::endl;
  std::cerr << "  min_z          : model bottom Z in mm" << std::endl;
  std::cerr << "  offset_mm      : contour offset in mm (default: 0)" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string png_dir = argv[1];
  const float layer_height = std::stof(argv[2]);
  const float min_z = std::stof(argv[3]);
  const float offset_mm = (argc > 4) ? std::stof(argv[4]) : 0.0f;

  // Collect and sort PNG files
  std::vector<std::string> png_files;
  for (const auto& entry : fs::directory_iterator(png_dir)) {
    if (entry.path().extension() == ".png") {
      png_files.push_back(entry.path().string());
    }
  }
  // Numeric sort: extract number from filename stem (e.g. "123.png" → 123)
  std::sort(png_files.begin(), png_files.end(), [](const std::string& a, const std::string& b) {
    auto num = [](const std::string& p) {
      return std::stoi(fs::path(p).stem().string());
    };
    return num(a) < num(b);
  });

  if (png_files.empty()) {
    std::cerr << "No PNG files found in " << png_dir << std::endl;
    return 1;
  }

  std::cout << "Found " << png_files.size() << " layer PNGs" << std::endl;

  // Load images
  std::vector<cv::Mat> layer_images;
  layer_images.reserve(png_files.size());
  for (const auto& path : png_files) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "Failed to read: " << path << std::endl;
      return 1;
    }
    layer_images.push_back(std::move(img));
  }

  // Config — display dims use Prusa SL1 defaults, bbox center at origin
  island::DetectionConfig config;
  config.display_width = 68.04f;
  config.display_height = 120.96f;
  config.layer_height = layer_height;
  config.model_bbox = {0, 0, min_z, 0, 0, 0};
  config.offset_mm = offset_mm;

  // Detect
  auto islands = island::detect_islands(layer_images, config);

  // Print results
  std::cout << "Detected " << islands.size() << " islands"
            << " (offset=" << offset_mm << "mm)" << std::endl;

  std::cout << std::fixed << std::setprecision(2);
  for (const auto& isl : islands) {
    std::cout << "  [" << isl.label << "] z=" << isl.z
              << "mm, pts=" << isl.contour.size();

    // Print first few contour points (world-space mm)
    int n = std::min(static_cast<int>(isl.contour.size()), 3);
    std::cout << "  contour=[";
    for (int j = 0; j < n; ++j) {
      if (j > 0) std::cout << ", ";
      std::cout << "(" << isl.contour[j].x << "," << isl.contour[j].y << ")";
    }
    if (static_cast<int>(isl.contour.size()) > n) std::cout << "...";
    std::cout << "]" << std::endl;
  }

  return 0;
}
