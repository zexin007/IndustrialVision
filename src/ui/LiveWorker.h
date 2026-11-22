#pragma once

#include "industrial_vision/ImageSequenceSource.h"

#include <QImage>
#include <QObject>
#include <QString>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

class LiveWorker : public QObject {
    Q_OBJECT

public:
    explicit LiveWorker(QObject* parent = nullptr);
    void requestStop() noexcept;

public slots:
    void process(QString folder, QString modelPath, int intervalMs);

signals:
    // QImage 自己持有像素后，才能安全地跨线程交给界面。
    void previewReady(QImage original, QImage marked, QString status);
    void finished(QString summary, bool fault);

private:
    std::atomic_bool stopRequested_{ false };

    std::mutex queueMutex_;
    std::condition_variable queueReady_;
    std::deque<ImageFrame> frames_;
    bool producerFinished_ = false;
    std::size_t peakDepth_ = 0;
};