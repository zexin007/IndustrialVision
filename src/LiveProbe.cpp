#include "industrial_vision/YoloInspector.h"
#include "industrial_vision/ImageSequenceSource.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

using Clock = std::chrono::steady_clock;

// 两个线程共享的有限队列。
// 采集线程放入 ImageFrame，检测线程取出 ImageFrame。
struct FrameQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<ImageFrame> frames;
    bool producerFinished = false;
    std::size_t peakDepth = 0;
};

int main()
{
    const std::filesystem::path imageFolder =
        LR"(T:\MyCode_qt\study\NEU-DET\val\images)";
    const std::filesystem::path modelPath =
        LR"(T:\MyCode_qt\study\models\NEU-DET-YOLOv8n\neu_det_yolov8n_best.onnx)";

    constexpr std::size_t queueCapacity = 3;
    //constexpr int frameIntervalMs = 20; // 先模拟每 100 ms 采一张
    // 以下是软件仿真的工位假设，不是 NEU-DET 的真实拍摄参数。
    constexpr double stripSpeedMmPerSec = 1000.0;
    constexpr double segmentLengthMm = 50.0;
    constexpr double cameraToMarkerMm = 500.0;
    constexpr double actionReserveMs = 50.0;

    constexpr int frameIntervalMs =
        static_cast<int>(segmentLengthMm / stripSpeedMmPerSec * 1000.0);

    constexpr double resultDeadlineMs =
        cameraToMarkerMm / stripSpeedMmPerSec * 1000.0 - actionReserveMs;

    ImageSequenceSource source;
    if (!source.open(imageFolder)) {
        std::cerr << "Cannot open image folder.\n";
        return 1;
    }

    try {
        // 模型只加载一次。之后只有检测线程调用 inspect()。
        YoloInspector inspector(modelPath);

        FrameQueue queue;
        std::atomic_bool stop{ false };

        std::size_t processed = 0;       // 仅检测线程修改
        std::size_t readFailures = 0;    // 仅采集线程修改
        std::size_t inspectFailures = 0; // 仅检测线程修改
        std::size_t overrunFrameId = 0;  // 仅采集线程修改

        std::exception_ptr producerError;
        std::exception_ptr consumerError;

        std::thread producer([&]() {
            try {
                auto nextTick = Clock::now();

                while (!stop.load()) {
                    // 按预定时间出帧，而不是“检测完一张才取下一张”。
                    std::this_thread::sleep_until(nextTick);
                    nextTick += std::chrono::milliseconds(frameIntervalMs);

                    if (stop.load()) {
                        break;
                    }

                    ImageFrame frame;
                    std::string readError;
                    if (!source.next(frame, readError)) {
                        if (readError.empty()) {
                            break; // 正常读到文件夹末尾
                        }

                        ++readFailures;
                        std::cerr << "Read failed, segment "
                            << frame.id << ": " << readError << '\n';
                        continue; // 坏图已被数据源跳过
                    }

                    frame.acquiredAt = Clock::now();

                    {
                        std::lock_guard<std::mutex> lock(queue.mutex);

                        if (queue.frames.size() == queueCapacity) {
                            // 不悄悄丢掉某段钢带的检测任务。
                            // 本版模拟工位进入故障停止状态。
                            overrunFrameId = frame.id;
                            stop.store(true);
                        }
                        else {
                            queue.frames.push_back(std::move(frame));
                            if (queue.frames.size() > queue.peakDepth) {
                                queue.peakDepth = queue.frames.size();
                            }
                        }
                    }

                    queue.ready.notify_one();

                    if (stop.load()) {
                        break;
                    }
                }
            }
            catch (...) {
                producerError = std::current_exception();
                stop.store(true);
            }

            {
                std::lock_guard<std::mutex> lock(queue.mutex);
                queue.producerFinished = true;
            }
            queue.ready.notify_all();
            });

        std::thread consumer([&]() {
            try {
                while (true) {
                    ImageFrame frame;

                    {
                        std::unique_lock<std::mutex> lock(queue.mutex);
                        queue.ready.wait(lock, [&]() {
                            return stop.load()
                                || !queue.frames.empty()
                                || queue.producerFinished;
                            });

                        //if (stop.load() || queue.frames.empty()) {
                        //    break;
                        //}

                        if (stop.load()) {
                            for (const auto& pending : queue.frames) {
                                std::cerr << "UNINSPECTED segment "
                                    << pending.id << ": stopped after overload\n";
                            }
                            queue.frames.clear();
                            break;
                        }

                        if (queue.frames.empty()) {
                            break; // 正常处理完队列中的所有帧
                        }

                        frame = std::move(queue.frames.front());
                        queue.frames.pop_front();
                    } // 解锁后再推理，不能拿着队列锁运行 YOLO。

                    const InspectionResult result =
                        inspector.inspect(frame.image);
                    const double totalMs =
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - frame.acquiredAt
                        ).count();
                    const double simulatedPositionMm =
                        (frame.id - 1) * segmentLengthMm;
                    const bool resultLate = totalMs > resultDeadlineMs;

                    ++processed;
                    if (!result.success) {
                        ++inspectFailures;
                        std::cerr << "Inspection failed, segment "
                            << frame.id << ": "
                            << result.errorMessage << '\n';
                    }

                    // 控制台输出也会占时间，所以只打印部分帧。
                    if (processed <= 10 || processed % 20 == 0) {
                        std::cout << "Segment " << frame.id
                            << " | boxes: "
                            << result.detections.size()
                            << " | inspect: "
                            << result.elapsedMs << " ms"
                            << " | acquire-to-result: "
                            << totalMs << " ms\n" << " | position: " << simulatedPositionMm << " mm"
                            << " | deadline: " << resultDeadlineMs << " ms"
                            << " | status: " << (resultLate ? "LATE" : "ON_TIME")
                            << '\n';
                    }
                }
            }
            catch (...) {
                consumerError = std::current_exception();
                stop.store(true);
                queue.ready.notify_all();
            }
            });

        producer.join();
        consumer.join();

        if (producerError) {
            std::rethrow_exception(producerError);
        }
        if (consumerError) {
            std::rethrow_exception(consumerError);
        }

        std::cout << "Processed: " << processed
            << " | read failures: " << readFailures
            << " | inspection failures: " << inspectFailures
            << " | peak queue depth: " << queue.peakDepth << '\n';

        if (overrunFrameId != 0) {
            std::cerr << "OVERLOAD: queue full at segment "
                << overrunFrameId << '\n';
            return 3;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "LiveProbe failed: " << e.what() << '\n';
        return 2;
    }

    return 0;
}