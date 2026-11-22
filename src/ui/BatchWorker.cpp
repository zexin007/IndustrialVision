#include "BatchWorker.h"

#include "industrial_vision/ImageSequenceSource.h"
#include "industrial_vision/TraditionalInspector.h"
#include "industrial_vision/YoloInspector.h"

#include <QElapsedTimer>

#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>

BatchWorker::BatchWorker(QObject* parent)
    : QObject(parent)
{
}

void BatchWorker::requestStop() noexcept
{
    // 界面线程只修改这个原子变量；工作线程在循环中读取。
    stopRequested_.store(true);
}

void BatchWorker::process(
    QString folder, int mode,
    QString modelPath, QString reportPath)
{
    int processed = 0;
    int failed = 0;
    int imagesWithDetections = 0;
    QString errorMessage;

    QElapsedTimer timer;
    timer.start();

    try {
        if (mode != 0 && mode != 1) {
            throw std::runtime_error("Unknown inspection mode.");
        }

        // 数据源属于本次批量任务，不改变界面的“下一张”位置。
        ImageSequenceSource source;
        if (!source.open(
            std::filesystem::path(folder.toStdWString()))) {
            throw std::runtime_error("Cannot open image folder.");
        }

        // 检测器在工作线程创建，并在整个批次中复用。
        std::unique_ptr<YoloInspector> yolo;
        if (mode == 1) {
            yolo = std::make_unique<YoloInspector>(
                std::filesystem::path(modelPath.toStdWString())
            );
        }

        std::ofstream report{
            std::filesystem::path(reportPath.toStdWString())
        };
        if (!report) {
            throw std::runtime_error("Cannot create CSV report.");
        }

        report << "filename,success,detection_count,inference_ms\n";

        while (!stopRequested_.load()) {
            ImageFrame frame;
            std::string readError;

            if (!source.next(frame, readError)) {
                if (readError.empty()) {
                    break;  // 正常到达末尾
                }

                ++processed;
                ++failed;
                report << frame.path.filename().string()
                    << ",0,0,0\n";
                emit progress(processed, failed,
                    imagesWithDetections);
                continue;
            }

            InspectionResult result =
                (mode == 0)
                ? inspectTraditional(frame.image)
                : yolo->inspect(frame.image);

            ++processed;

            if (!result.success) {
                ++failed;
            }
            else if (!result.detections.empty()) {
                ++imagesWithDetections;
            }

            report << frame.path.filename().string() << ','
                << (result.success ? 1 : 0) << ','
                << result.detections.size() << ','
                << result.elapsedMs << '\n';

            emit progress(processed, failed,
                imagesWithDetections);
        }

        report.flush();
        if (!report) {
            throw std::runtime_error("Failed to write CSV report.");
        }
    }
    catch (const std::exception& e) {
        errorMessage = QString::fromUtf8(e.what());
    }

    // 无论正常结束、停止还是发生错误，都只发出一次完成信号。
    emit finished(processed, failed, imagesWithDetections,
        timer.elapsed(), stopRequested_.load(),
        errorMessage);
}