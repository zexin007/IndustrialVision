#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    std::optional<int> classId;
    std::optional<float> confidence;
};

struct InspectionResult {
    bool success = false;
    std::vector<Detection> detections;
    double elapsedMs = 0.0;
    std::string errorMessage;
};