#include "industrial_vision/YoloInspector.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

struct YoloInspector::Impl {
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "yolo_inspector" };
    Ort::SessionOptions options;
    Ort::Session session;

    explicit Impl(const std::filesystem::path& modelPath)
        : session(env, modelPath.c_str(), options)
    {
    }
};

YoloInspector::YoloInspector(const std::filesystem::path& modelPath)
    : impl_(std::make_unique<Impl>(modelPath))
{
}

YoloInspector::~YoloInspector() = default;

InspectionResult YoloInspector::inspect(const cv::Mat& image)
{
    InspectionResult result;

    if (image.empty() || image.type() != CV_8UC3) {
        result.errorMessage = "Expected a non-empty 8-bit BGR image.";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();

    constexpr int inputSize = 640;
    constexpr int classCount = 6;
    constexpr float scoreThreshold = 0.25f;
    constexpr float nmsThreshold = 0.45f;

    try {
        // 1. BGR -> RGB、缩放、归一化、HWC -> NCHW。
        cv::Mat blob = cv::dnn::blobFromImage(
            image, 1.0 / 255.0,
            cv::Size(inputSize, inputSize),
            cv::Scalar(), true, false, CV_32F
        );

        // 2. 为本次推理准备输入张量。
        // inputTensor 使用 blob 的数据；Run 结束前 blob 必须存在。
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName =
            impl_->session.GetInputNameAllocated(0, allocator);
        auto outputName =
            impl_->session.GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };

        const std::array<int64_t, 4> inputShape{
            1, 3, inputSize, inputSize
        };
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            reinterpret_cast<float*>(blob.data),
            blob.total(),
            inputShape.data(),
            inputShape.size()
        );

        // 3. 复用构造时加载的 session 执行推理。
        auto outputs = impl_->session.Run(
            Ort::RunOptions{ nullptr },
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("Expected one output tensor.");
        }

        const auto shape =
            outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // 当前模型输出应为 [1, 10, 8400]。
        if (shape.size() != 3 ||
            shape[0] != 1 ||
            shape[1] != 4 + classCount ||
            shape[2] <= 0) {
            throw std::runtime_error("Unexpected YOLO output shape.");
        }

        const float* raw = outputs[0].GetTensorData<float>();
        const std::size_t count =
            static_cast<std::size_t>(shape[2]);

        // 4. 按类别收集超过分数阈值的候选框。
        std::array<std::vector<cv::Rect>, classCount> boxesByClass;
        std::array<std::vector<float>, classCount> scoresByClass;

        const float scaleX =
            static_cast<float>(image.cols) / inputSize;
        const float scaleY =
            static_cast<float>(image.rows) / inputSize;

        for (std::size_t i = 0; i < count; ++i) {
            float bestScore = 0.0f;
            int bestClass = -1;

            for (int cls = 0; cls < classCount; ++cls) {
                const float score = raw[(4 + cls) * count + i];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = cls;
                }
            }

            if (bestClass < 0 ||
                bestScore < scoreThreshold) {
                continue;
            }

            const float cx = raw[0 * count + i];
            const float cy = raw[1 * count + i];
            const float w = raw[2 * count + i];
            const float h = raw[3 * count + i];

            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(w) || !std::isfinite(h) ||
                w <= 0 || h <= 0) {
                continue;
            }

            // 5. 将 640×640 上的框换算到原图，并裁到图像范围内。
            const int x1 = std::clamp(
                cvRound((cx - w / 2.0f) * scaleX),
                0, image.cols
            );
            const int y1 = std::clamp(
                cvRound((cy - h / 2.0f) * scaleY),
                0, image.rows
            );
            const int x2 = std::clamp(
                cvRound((cx + w / 2.0f) * scaleX),
                0, image.cols
            );
            const int y2 = std::clamp(
                cvRound((cy + h / 2.0f) * scaleY),
                0, image.rows
            );

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            boxesByClass[bestClass].emplace_back(
                x1, y1, x2 - x1, y2 - y1
            );
            scoresByClass[bestClass].push_back(bestScore);
        }

        // 6. 每个类别分别做 NMS，保存最终检测结果。
        for (int cls = 0; cls < classCount; ++cls) {
            std::vector<int> keptIndices;

            cv::dnn::NMSBoxes(
                boxesByClass[cls],
                scoresByClass[cls],
                scoreThreshold,
                nmsThreshold,
                keptIndices
            );

            for (int index : keptIndices) {
                result.detections.push_back(Detection{
                    boxesByClass[cls][index],
                    cls,
                    scoresByClass[cls][index]
                    });
            }
        }

        // 没有检测框也是一次成功的推理。
        result.success = true;
    }
    catch (const Ort::Exception& e) {
        result.errorMessage = e.what();
    }
    catch (const cv::Exception& e) {
        result.errorMessage = e.what();
    }
    catch (const std::exception& e) {
        result.errorMessage = e.what();
    }

    result.elapsedMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();

    return result;
}