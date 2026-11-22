#include "LiveWorker.h"

#include "industrial_vision/YoloInspector.h"

#include <QImage>
#include <QStringList>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using Clock = std::chrono::steady_clock;

LiveWorker::LiveWorker(QObject* parent)
    : QObject(parent)
{
}

void LiveWorker::requestStop() noexcept
{
    stopRequested_.store(true);
    queueReady_.notify_all(); // 唤醒可能正在等图的检测线程。
}

void LiveWorker::process(
    QString folder, QString modelPath, int intervalMs)
{
    constexpr std::size_t queueCapacity = 3;
    constexpr double segmentLengthMm = 50.0;
    constexpr double stripSpeedMmPerSec = 500.0;
    constexpr double cameraToMarkerMm = 500.0;
    constexpr double actionReserveMs = 50.0;

    constexpr double deadlineMs =
        cameraToMarkerMm / stripSpeedMmPerSec * 1000.0
        - actionReserveMs;

    std::size_t processed = 0;
    std::size_t readFailures = 0;    // 采集线程写；join 后读取。
    std::size_t inspectFailures = 0;
    std::size_t lateCount = 0;
    std::size_t overrunId = 0;       // 采集线程写；join 后读取。
    QStringList uninspectedIds;

    QString error;
    std::exception_ptr producerError;
    std::thread producer;

    try {
        if (intervalMs <= 0) {
            throw std::runtime_error("Frame interval must be positive.");
        }

        // 本次连续任务使用独立图片源，不改变“下一张”的位置。
        ImageSequenceSource source;
        if (!source.open(
            std::filesystem::path(folder.toStdWString()))) {
            throw std::runtime_error("Cannot open image folder.");
        }

        // 在后台线程加载模型；一轮任务只创建一次。
        YoloInspector inspector(
            std::filesystem::path(modelPath.toStdWString()));

        producer = std::thread([&, this]() {
            try {
                auto nextTick = Clock::now();

                while (!stopRequested_.load()) {
                    std::this_thread::sleep_until(nextTick);
                    nextTick += std::chrono::milliseconds(intervalMs);

                    if (stopRequested_.load()) {
                        break;
                    }

                    ImageFrame frame;
                    std::string readError;
                    if (!source.next(frame, readError)) {
                        if (readError.empty()) {
                            break; // 正常读完
                        }
                        ++readFailures;
                        continue; // 坏图由数据源跳过
                    }

                    frame.acquiredAt = Clock::now();

                    {
                        std::lock_guard<std::mutex> lock(queueMutex_);
                        if (frames_.size() == queueCapacity) {
                            overrunId = frame.id;
                            stopRequested_.store(true);
                        }
                        else {
                            frames_.push_back(std::move(frame));
                            if (frames_.size() > peakDepth_) {
                                peakDepth_ = frames_.size();
                            }
                        }
                    }

                    queueReady_.notify_one();

                    if (stopRequested_.load()) {
                        break;
                    }
                }
            }
            catch (...) {
                producerError = std::current_exception();
                stopRequested_.store(true);
            }

            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                producerFinished_ = true;
            }
            queueReady_.notify_all();
            });

        // 当前 Qt 工作线程充当消费者，不能在这里操作 QWidget。
        auto lastPreview = Clock::now() - std::chrono::seconds(1);

        while (true) {
            ImageFrame frame;

            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                queueReady_.wait(lock, [this]() {
                    return stopRequested_.load()
                        || !frames_.empty()
                        || producerFinished_;
                    });

                if (stopRequested_.load()) {
                    for (const auto& pending : frames_) {
                        uninspectedIds.append(
                            QString::number(pending.id));
                    }
                    frames_.clear();
                    break;
                }

                if (frames_.empty()) {
                    break; // 采集结束，队列也处理完。
                }

                frame = std::move(frames_.front());
                frames_.pop_front();
            } // 先解锁，再进行耗时推理。

            const InspectionResult result =
                inspector.inspect(frame.image);

            const double totalMs =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - frame.acquiredAt
                ).count();

            ++processed;
            if (!result.success) {
                ++inspectFailures;
            }
            if (totalMs > deadlineMs) {
                ++lateCount;
            }

            // 界面最多约每 200 ms 更新一次；
            // 检测仍对取出的每帧执行，不因界面限频而跳过检测。
            const auto now = Clock::now();
            if (now - lastPreview < std::chrono::milliseconds(200)) {
                continue;
            }
            lastPreview = now;

            cv::Mat marked = frame.image.clone();
            if (result.success) {
                for (const auto& detection : result.detections) {
                    cv::rectangle(
                        marked, detection.box,
                        cv::Scalar(0, 0, 255), 2);
                }
            }

            cv::Mat originalRgb;
            cv::Mat markedRgb;
            cv::cvtColor(frame.image, originalRgb, cv::COLOR_BGR2RGB);
            cv::cvtColor(marked, markedRgb, cv::COLOR_BGR2RGB);

            // copy()：Qt 图像不再依赖下面两个局部 Mat 的像素。
            QImage original(
                originalRgb.data, originalRgb.cols, originalRgb.rows,
                static_cast<qsizetype>(originalRgb.step),
                QImage::Format_RGB888);
            QImage resultImage(
                markedRgb.data, markedRgb.cols, markedRgb.rows,
                static_cast<qsizetype>(markedRgb.step),
                QImage::Format_RGB888);

            const QString state =
                !result.success ? "ERROR"
                : totalMs > deadlineMs ? "LATE"
                : result.detections.empty() ? "NO_CANDIDATE"
                : "CANDIDATE";

            const double positionMm =
                (frame.id - 1) * segmentLengthMm;

            const QString status =
                QString("段号：%1 | 模拟位置：%2 mm | 候选框：%3"
                    " | 总耗时：%4 ms | %5")
                .arg(static_cast<qulonglong>(frame.id))
                .arg(positionMm, 0, 'f', 0)
                .arg(static_cast<qulonglong>(
                    result.detections.size()))
                .arg(totalMs, 0, 'f', 1)
                .arg(state);

            emit previewReady(
                original.copy(), resultImage.copy(), status);
        }
    }
    catch (const std::exception& e) {
        error = QString::fromUtf8(e.what());
        requestStop();
    }

    // 不论正常结束、手动停止还是异常，都等采集线程退出。
    if (producer.joinable()) {
        producer.join();
    }

    if (producerError) {
        try {
            std::rethrow_exception(producerError);
        }
        catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        }
    }

    const bool fault = !error.isEmpty() || overrunId != 0;

    QString summary =
        QString("连续检测结束 | 已检：%1 | 读图失败：%2"
            " | 检测失败：%3 | 超期：%4 | 队列峰值：%5")
        .arg(static_cast<qulonglong>(processed))
        .arg(static_cast<qulonglong>(readFailures))
        .arg(static_cast<qulonglong>(inspectFailures))
        .arg(static_cast<qulonglong>(lateCount))
        .arg(static_cast<qulonglong>(peakDepth_));

    if (!uninspectedIds.isEmpty()) {
        summary += " | 未检段号：" + uninspectedIds.join(",");
    }
    if (overrunId != 0) {
        summary += QString(" | 过载段号：%1").arg(
            static_cast<qulonglong>(overrunId));
    }
    if (!error.isEmpty()) {
        summary += " | 错误：" + error;
    }

    emit finished(summary, fault);
}