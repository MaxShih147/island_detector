#include "island_detector.h"

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

static void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " <png_dir> <layer_height_mm> <min_z>"
            << std::endl;
  std::cerr << "  png_dir        : directory containing layer PNGs (sorted by name)"
            << std::endl;
  std::cerr << "  layer_height_mm: layer height in mm"
            << std::endl;
  std::cerr << "  min_z          : model bottom Z in mm"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string png_dir = argv[1];
  const float layer_height = std::stof(argv[2]);
  const float min_z = std::stof(argv[3]);

  // Collect and sort PNG files
  std::vector<std::string> png_files;
  for (const auto& entry : fs::directory_iterator(png_dir)) {
    if (entry.path().extension() == ".png") {
      png_files.push_back(entry.path().string());
    }
  }
  std::sort(png_files.begin(), png_files.end());

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

  // Config — display dims use Prusa SL1 defaults, bbox only needs min_z
  island::DetectionConfig config;
  config.display_width = 68.04f;
  config.display_height = 120.96f;
  config.layer_height = layer_height;
  config.model_bbox = {0, 0, min_z, 0, 0, 0};

  // Detect
  auto islands = island::detect_islands(layer_images, config);

  // Print results
  std::cout << "Detected " << islands.size() << " islands" << std::endl;
  for (const auto& isl : islands) {
    std::cout << "  [" << isl.label << "] z=" << isl.z
              << "mm, contour_pts=" << isl.contour.size()
              << std::endl;
  }

  return 0;
}
