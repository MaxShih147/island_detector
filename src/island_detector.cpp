#include "island_detector.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace island {

/// Binarize image: threshold at 127 → 0 or 255.
static cv::Mat binarize(const cv::Mat& img) {
  cv::Mat bin;
  if (img.channels() > 1) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin, 127, 255, cv::THRESH_BINARY);
  }
  else {
    cv::threshold(img, bin, 127, 255, cv::THRESH_BINARY);
  }
  return bin;
}

std::vector<Island> detect_islands(
    const std::vector<cv::Mat>& layer_images,
    const DetectionConfig& config) {

  std::vector<Island> all_islands;

  if (layer_images.size() < 2) {
    return all_islands;
  }

  const float base_z = config.model_bbox.min_z;

  // Process consecutive pairs: compare layer i (current) against layer i-1 (below)
  for (size_t i = 1; i < layer_images.size(); ++i) {
    cv::Mat img_below = binarize(layer_images[i - 1]);
    cv::Mat img_current = binarize(layer_images[i]);

    const float z = base_z + static_cast<float>(i) * config.layer_height;

    // Connected components on current layer
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        img_current, labels, stats, centroids, 8, CV_32S);

    // Check each component (skip label 0 = background)
    for (int lbl = 1; lbl < num_labels; ++lbl) {
      int left   = stats.at<int>(lbl, cv::CC_STAT_LEFT);
      int top    = stats.at<int>(lbl, cv::CC_STAT_TOP);
      int width  = stats.at<int>(lbl, cv::CC_STAT_WIDTH);
      int height = stats.at<int>(lbl, cv::CC_STAT_HEIGHT);

      // Check overlap with layer below within bounding box
      bool has_overlap = false;
      for (int y = top; y < top + height && !has_overlap; ++y) {
        for (int x = left; x < left + width && !has_overlap; ++x) {
          if (labels.at<int>(y, x) == lbl && img_below.at<uint8_t>(y, x) > 0) {
            has_overlap = true;
          }
        }
      }

      if (has_overlap) continue;

      // --- Island found: extract contour ---

      // Build binary mask for this component
      cv::Mat mask = cv::Mat::zeros(img_current.rows, img_current.cols, CV_8UC1);
      for (int y = top; y < top + height; ++y) {
        for (int x = left; x < left + width; ++x) {
          if (labels.at<int>(y, x) == lbl) {
            mask.at<uint8_t>(y, x) = 255;
          }
        }
      }

      // Find contours
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      if (contours.empty()) continue;

      // Pick the largest contour by point count
      size_t best_idx = 0;
      for (size_t ci = 1; ci < contours.size(); ++ci) {
        if (contours[ci].size() > contours[best_idx].size()) {
          best_idx = ci;
        }
      }

      // Convert cv::Point (int) → Point2f
      const auto& best = contours[best_idx];
      std::vector<Point2f> contour_pts;
      contour_pts.reserve(best.size());
      for (const auto& pt : best) {
        contour_pts.push_back({static_cast<float>(pt.x), static_cast<float>(pt.y)});
      }

      Island island;
      island.label = -1;  // assigned below after sorting
      island.contour = std::move(contour_pts);
      island.z = z;

      all_islands.push_back(std::move(island));
    }
  }

  // Sort by z ascending, then assign global labels 0..N-1
  std::sort(all_islands.begin(), all_islands.end(),
            [](const Island& a, const Island& b) { return a.z < b.z; });

  for (int i = 0; i < static_cast<int>(all_islands.size()); ++i) {
    all_islands[i].label = i;
  }

  return all_islands;
}

}  // namespace island
