#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "industrial_vision/InspectionResult.h"

//struct InspectionResult {
//    bool success = false;
//    std::vector<cv::Rect> boxes;
//    double elapsedMs = 0.0;
//    std::string errorMessage;
//};

// 只使用原图像素，不读取 GT 掩码。
InspectionResult inspectTraditional(const cv::Mat& image);