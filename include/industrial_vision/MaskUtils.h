#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace industrial_vision {

std::vector<cv::Rect> extractDefectBoxes(const cv::Mat& mask);

}  // namespace industrial_vision
