#include "industrial_vision/TraditionalInspector.h"

#include <opencv2/imgproc.hpp>
#include <chrono>

InspectionResult inspectTraditional(const cv::Mat& image)
{
    InspectionResult result;

    if (image.empty() || image.type() != CV_8UC3) {
        result.errorMessage = "Expected a non-empty 8-bit BGR image.";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();

    try {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 黑帽操作突出局部较暗的区域。尺寸和阈值只是基线参数。
        cv::Mat response;
        const cv::Mat largeKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(15, 15));
        cv::morphologyEx(
            gray, response, cv::MORPH_BLACKHAT, largeKernel);

        cv::Mat binary;
        cv::threshold(
            response, binary, 25, 255, cv::THRESH_BINARY);

        // 去掉一些零散亮点。
        const cv::Mat smallKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(
            binary, binary, cv::MORPH_OPEN, smallKernel);

        cv::Mat labels, stats, centroids;
        const int count = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids, 8, CV_32S);

        // 编号 0 是背景，从 1 开始才是候选区域。
        for (int i = 1; i < count; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < 12 || area > 2000) {
                continue;
            }

            result.detections.emplace_back(
                Detection{
                cv::Rect(
                    stats.at<int>(i, cv::CC_STAT_LEFT),
                    stats.at<int>(i, cv::CC_STAT_TOP),
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT)
                ),
                std::nullopt,
                std::nullopt
            });
        }

        result.success = true;
    }
    catch (const cv::Exception& e) {
        result.errorMessage = e.what();
    }

    const auto end = std::chrono::steady_clock::now();
    result.elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}