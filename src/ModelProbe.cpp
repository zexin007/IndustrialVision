#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>  // std::clamp
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <fstream>

int main()
{
    // 1. 创建 ONNX Runtime 环境，并加载模型。
    // session 创建一次即可；以后连续处理图片时应复用它。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_probe");
    Ort::SessionOptions options;

    Ort::Session session(
        env,
        LR"(T:\MyCode_qt\study\models\NEU-DET-YOLOv8n\neu_det_yolov8n_best.onnx)",
        options
    );

    Ort::AllocatorWithDefaultOptions allocator;

    // 2. 读取原图。后面会在这张图上画框并保存。
    cv::Mat image = cv::imread(
        R"(T:\MyCode_qt\study\NEU-DET\val\images\crazing_271.jpg)"
    );
    if (image.empty()) {
        std::cerr << "Cannot read image\n";
        return 2;
    }

    // 3. 预处理：缩放到 640×640、像素除以 255、
    // BGR 转 RGB，并整理成模型需要的 NCHW 浮点数据。
    cv::Mat blob = cv::dnn::blobFromImage(
        image,
        1.0 / 255.0,
        cv::Size(640, 640),
        cv::Scalar(),
        true,   // swapRB：BGR -> RGB
        false,  // 不裁剪
        CV_32F
    );

    // 4. 从模型取得输入和输出名称。
    // inputName、outputName 会管理名称所占的内存。
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    // 5. 创建输入张量。这里的张量使用 blob 的像素数据，
    // 所以在 session.Run() 完成前，blob 必须保持有效。
    std::array<int64_t, 4> inputShape{ 1, 3, 640, 640 };
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

    const char* inputNames[] = { inputName.get() };
    const char* outputNames[] = { outputName.get() };

    // 6. 执行一次模型推理。
    auto outputs = session.Run(
        Ort::RunOptions{ nullptr },
        inputNames, &inputTensor, 1,
        outputNames, 1
    );

    // 7. 检查实际输出。这个模型预期是 [1, 10, 8400]：
    // 4 个框参数 + 6 个类别分数，共 8400 个候选点。
    const auto outputShape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Actual output shape:";
    for (auto dim : outputShape) {
        std::cout << ' ' << dim;
    }
    std::cout << '\n';

    if (outputShape.size() != 3 ||
        outputShape[0] != 1 ||
        outputShape[1] != 10 ||
        outputShape[2] <= 0) {
        std::cerr << "Unexpected model output shape\n";
        return 3;
    }

    const float* raw = outputs[0].GetTensorData<float>();
    const std::size_t count =
        static_cast<std::size_t>(outputShape[2]);

    //// 8. 在所有候选点的 6 个类别分数中，找出最高分。
    //// 输出排布是 [通道][候选点]，因此下标为：
    //// 通道编号 * count + 候选点编号。
    //float bestScore = 0.0f;
    //std::size_t bestIndex = 0;
    //int bestClass = -1;

    //for (std::size_t i = 0; i < count; ++i) {
    //    for (int cls = 0; cls < 6; ++cls) {
    //        const float score = raw[(4 + cls) * count + i];

    //        if (score > bestScore) {
    //            bestScore = score;
    //            bestIndex = i;
    //            bestClass = cls;
    //        }
    //    }
    //}

    //if (bestClass < 0) {
    //    std::cerr << "No candidate found\n";
    //    return 4;
    //}

    //std::cout << "Best candidate: " << bestIndex
    //    << ", class: " << bestClass
    //    << ", score: " << bestScore << '\n';

    //cv::Mat prediction = image.clone();
    //cv::Mat groundTruth = image.clone();
    //// 9. 取出最高分候选框。模型给的是 640×640 图像上的
    //// 中心点 x/y 和框的宽/高。
    //const float cx = raw[0 * count + bestIndex];
    //const float cy = raw[1 * count + bestIndex];
    //const float w = raw[2 * count + bestIndex];
    //const float h = raw[3 * count + bestIndex];

    //std::cout << "Box center x/y, width/height: "
    //    << cx << ", " << cy << ", "
    //    << w << ", " << h << '\n';

    //// 10. 将模型坐标映射回原图坐标。
    //// 原图若为 200×200，两个比例都是 200/640。
    //const float scaleX = static_cast<float>(image.cols) / 640.0f;
    //const float scaleY = static_cast<float>(image.rows) / 640.0f;

    //// 中心点 ± 半个宽高，得到左上角与右下角。
    //// clamp 防止坐标超出图像边界。
    //const int x1 = std::clamp(
    //    cvRound((cx - w / 2.0f) * scaleX), 0, image.cols);
    //const int y1 = std::clamp(
    //    cvRound((cy - h / 2.0f) * scaleY), 0, image.rows);
    //const int x2 = std::clamp(
    //    cvRound((cx + w / 2.0f) * scaleX), 0, image.cols);
    //const int y2 = std::clamp(
    //    cvRound((cy + h / 2.0f) * scaleY), 0, image.rows);

    //if (x2 <= x1 || y2 <= y1) {
    //    std::cerr << "Invalid box\n";
    //    return 5;
    //}
    //// 11. OpenCV 的 Rect 使用 x、y、宽、高，
    //// 因此宽高要用两个角的坐标相减。
    //cv::rectangle(
    //    prediction,
    //    cv::Rect(x1, y1, x2 - x1, y2 - y1),
    //    cv::Scalar(0, 0, 255),  // OpenCV 的红色：BGR
    //    2
    //);

    // 分数阈值决定哪些候选框进入后处理。
// NMS 阈值决定两个重叠框达到多大程度时只保留高分框。
    constexpr float scoreThreshold = 0.25f;
    constexpr float nmsThreshold = 0.45f;

    // 按类别分别保存框和对应分数。
    // 同一个下标的 box 和 score 属于同一个候选点。
    std::array<std::vector<cv::Rect>, 6> boxesByClass;
    std::array<std::vector<float>, 6> scoresByClass;

    const float scaleX = static_cast<float>(image.cols) / 640.0f;
    const float scaleY = static_cast<float>(image.rows) / 640.0f;

    // 遍历全部 8400 个候选点。
    for (std::size_t i = 0; i < count; ++i) {
        // 找出这个候选点分数最高的类别。
        float score = 0.0f;
        int classId = -1;

        for (int cls = 0; cls < 6; ++cls) {
            const float classScore = raw[(4 + cls) * count + i];
            if (classScore > score) {
                score = classScore;
                classId = cls;
            }
        }

        // 低分候选点直接丢弃，不参与 NMS。
        if (classId < 0 || score < scoreThreshold) {
            continue;
        }

        // 模型输出的是 640×640 上的中心点和宽高。
        const float cx = raw[0 * count + i];
        const float cy = raw[1 * count + i];
        const float w = raw[2 * count + i];
        const float h = raw[3 * count + i];

        // 换算为原图上的两个角，并裁到图像范围内。
        const int x1 = std::clamp(
            cvRound((cx - w / 2.0f) * scaleX), 0, image.cols);
        const int y1 = std::clamp(
            cvRound((cy - h / 2.0f) * scaleY), 0, image.rows);
        const int x2 = std::clamp(
            cvRound((cx + w / 2.0f) * scaleX), 0, image.cols);
        const int y2 = std::clamp(
            cvRound((cy + h / 2.0f) * scaleY), 0, image.rows);

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        boxesByClass[classId].emplace_back(
            x1, y1, x2 - x1, y2 - y1
        );
        scoresByClass[classId].push_back(score);
    }

    // 在原图副本上画预测结果。
    cv::Mat prediction = image.clone();
    cv::Mat groundTruth = image.clone();

    int candidateCount = 0;
    int keptCount = 0;

    for (int cls = 0; cls < 6; ++cls) {
        candidateCount += static_cast<int>(boxesByClass[cls].size());

        // 每个类别分别做 NMS，返回该类别中要保留的下标。
        std::vector<int> keptIndices;
        cv::dnn::NMSBoxes(
            boxesByClass[cls],
            scoresByClass[cls],
            scoreThreshold,
            nmsThreshold,
            keptIndices
        );

        for (int index : keptIndices) {
            cv::rectangle(
                prediction,
                boxesByClass[cls][index],
                cv::Scalar(0, 0, 255),
                2
            );
            ++keptCount;
        }
    }

    std::cout << "Candidates after score filter: "
        << candidateCount << '\n';
    std::cout << "Detections after NMS: "
        << keptCount << '\n';

    std::ifstream labels(
        R"(T:\MyCode_qt\study\NEU-DET\val\labels\crazing_271.txt)"
    );
    if (!labels) {
        std::cerr << "Cannot read label file\n";
        return 6;
    }

    int classId;
    float gx, gy, gw, gh;

    // 每行：类别 中心x 中心y 宽 高；坐标都除以了图像宽高。
    while (labels >> classId >> gx >> gy >> gw >> gh) {
        cv::Rect gtBox(
            cvRound((gx - gw / 2.0f) * image.cols),
            cvRound((gy - gh / 2.0f) * image.rows),
            cvRound(gw * image.cols),
            cvRound(gh * image.rows)
        );

        gtBox &= cv::Rect(0, 0, image.cols, image.rows);
        if (gtBox.area() > 0) {
            cv::rectangle(
                groundTruth, gtBox, cv::Scalar(0, 255, 0), 2
            );
        }
    }

    

    // 12. 保存结果图，并检查是否保存成功。
    const std::string outputPath =
        R"(T:\MyCode_qt\study\IndustrialVision\out\best_candidate.png)";

    if (!cv::imwrite(outputPath, image)) {
        std::cerr << "Cannot save result image\n";
        return 6;
    }

    cv::Mat comparison;
    cv::hconcat(prediction, groundTruth, comparison);

    if (!cv::imwrite(outputPath, comparison)) {
        std::cerr << "Cannot save comparison image\n";
        return 7;
    }


    std::cout << "Saved: " << outputPath << '\n';
    return 0;
}