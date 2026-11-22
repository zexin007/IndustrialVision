#pragma once

#include <QObject>
#include <QString>
#include <atomic>

class BatchWorker : public QObject {
    Q_OBJECT

public:
    explicit BatchWorker(QObject* parent = nullptr);

    // 界面线程可以调用它：只修改原子变量。
    void requestStop() noexcept;

public slots:
    void process(QString folder, int mode,
        QString modelPath, QString reportPath);

signals:
    void progress(int processed, int failed, int imagesWithDetections);

    void finished(int processed, int failed,
        int imagesWithDetections,
        qint64 totalMs, bool stopped,
        QString errorMessage);

private:
    std::atomic_bool stopRequested_{ false };
};