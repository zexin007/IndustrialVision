#include "industrial_vision/MaskUtils.h"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

namespace industrial_vision {

std::vector<cv::Rect> extractDefectBoxes(const cv::Mat& mask)
{
    if (mask.empty() || mask.channels() != 1) {
        return {};
    }

    cv::Mat binary;
    cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        binary,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    std::vector<cv::Rect> boxes;
    boxes.reserve(contours.size());

    for (const auto& contour : contours) {
        boxes.push_back(cv::boundingRect(contour));
    }

    return boxes;
}

}  // namespace industrial_vision
