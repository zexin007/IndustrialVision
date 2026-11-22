#pragma once

#include "industrial_vision/InspectionResult.h"

#include <opencv2/core.hpp>
#include <filesystem>
#include <memory>

class YoloInspector {
public:
    explicit YoloInspector(const std::filesystem::path& modelPath);
    ~YoloInspector();

    //禁止用operator=已有的一个检测器对象
    YoloInspector(const YoloInspector&) = delete;
    YoloInspector& operator=(const YoloInspector&) = delete;

    InspectionResult inspect(const cv::Mat& image);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};