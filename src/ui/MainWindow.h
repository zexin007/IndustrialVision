#pragma once

#include "industrial_vision/ImageSequenceSource.h"
#include <QWidget>
#include "industrial_vision/YoloInspector.h"
#include <memory>

#include <QThread>

class QComboBox;
class QLabel;
class QPushButton;

class BatchWorker;

class MainWindow : public QWidget {
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    void openFolder();
    void showNextFrame();

    void runBatch();

    std::filesystem::path currentFolder_;
    QPushButton* batchButton_ = nullptr;


    // 数据源是窗口的成员。按钮点击结束后，它仍保存当前读取位置。
    ImageSequenceSource source_;

    // 控件由 Qt 的父对象机制管理；这些指针用于后续访问控件。
    QPushButton* nextButton_ = nullptr;
    QLabel* imageLabel_ = nullptr;
    QLabel* fileLabel_ = nullptr;
    QLabel* originalLabel_ = nullptr;

    QComboBox* modeCombo_ = nullptr;
    std::unique_ptr<YoloInspector> yoloInspector_;

	// 批处理线程和工作对象
    QThread batchThread_;
    BatchWorker* batchWorker_ = nullptr;
    QPushButton* stopButton_ = nullptr;
};