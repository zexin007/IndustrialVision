#include "MainWindow.h"
#include "BatchWorker.h"

#include "industrial_vision/TraditionalInspector.h"
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <string>

#include <QComboBox>
#include <exception>

#include <QElapsedTimer>
#include <fstream>

MainWindow::MainWindow(QWidget* parent)
    : QWidget(parent)
{
    setWindowTitle("Industrial Vision");
    resize(700, 800);

    // this 是父窗口。Qt 会管理这些控件的生命周期。
    auto* layout = new QVBoxLayout(this);
    auto* openButton = new QPushButton("选择图片文件夹", this);
    modeCombo_ = new QComboBox(this);
    modeCombo_->addItems({ "传统视觉", "YOLO" });
    layout->addWidget(modeCombo_);

    nextButton_ = new QPushButton("下一张", this);
    //imageLabel_ = new QLabel("请选择文件夹", this);
    originalLabel_ = new QLabel("原图", this);
    imageLabel_ = new QLabel("检测结果", this);
    batchButton_ = new QPushButton("批量检测文件夹", this);
    layout->addWidget(batchButton_);
    
    //创建按钮，并把 this（窗口）设为它的父对象，由 Qt 管理按钮的生命周期。
    // 初始禁用，因为还没有批量任务可停止；
    // `addWidget` 把它放进布局。
    stopButton_ = new QPushButton("停止批量检测", this);
    stopButton_->setEnabled(false);
    layout->addWidget(stopButton_);

    connect(stopButton_, &QPushButton::clicked, this, [this]() {
        if (batchThread_.isRunning() && batchWorker_) {
            batchWorker_->requestStop();
            stopButton_->setEnabled(false);
            fileLabel_->setText("正在停止，等待当前图片处理完成...");
        }
        });

    connect(&batchThread_, &QThread::finished, this, [this]() {
        batchWorker_ = nullptr;
        batchButton_->setEnabled(true);
        stopButton_->setEnabled(false);
        });
    


    fileLabel_ = new QLabel(this);

    originalLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setAlignment(Qt::AlignCenter);

    nextButton_->setEnabled(false);

    layout->addWidget(openButton);
    layout->addWidget(nextButton_);
    
    //layout->addWidget(imageLabel_);
    auto* imageRow = new QHBoxLayout;
    imageRow->addWidget(originalLabel_);
    imageRow->addWidget(imageLabel_);
    layout->addLayout(imageRow);

    layout->addWidget(fileLabel_);

    // 点击按钮时，调用窗口自己的成员函数。
    connect(openButton, &QPushButton::clicked,
        this, [this]() { openFolder(); });

    connect(nextButton_, &QPushButton::clicked,
        this, [this]() { showNextFrame(); });
    connect(batchButton_, &QPushButton::clicked,
        this, [this]() { runBatch(); });
}

MainWindow::~MainWindow()
{
    if (batchThread_.isRunning()) {
        if (batchWorker_) {
            batchWorker_->requestStop();
        }
        batchThread_.quit();
        batchThread_.wait();
    }
}
void MainWindow::openFolder()
{
    const QString folder =
        QFileDialog::getExistingDirectory(this, "选择图片文件夹");

    if (folder.isEmpty()) {
        return;
    }

    if (!source_.open(std::filesystem::path(folder.toStdWString()))) {
        nextButton_->setEnabled(false);
        imageLabel_->setText("无法打开图片文件夹");
        fileLabel_->clear();
        QMessageBox::warning(this, "打开失败",
            "文件夹中没有可读取的原图");
        return;
    }
    currentFolder_ = std::filesystem::path(folder.toStdWString());
    nextButton_->setEnabled(true);
    showNextFrame();  // 选择文件夹后立即显示第一张。
}

void MainWindow::showNextFrame()
{
    ImageFrame frame;
    std::string errorMessage;

    if (!source_.next(frame, errorMessage)) {
        if (errorMessage.empty()) {
            fileLabel_->setText("已到文件夹末尾");
            nextButton_->setEnabled(false);
        }
        else {
            // 坏图已被数据源跳过；用户可继续点击读取下一张。
            QMessageBox::warning(
                this, "读取失败",
                QString::fromStdString(errorMessage));
        }
        return;
    }

    // OpenCV 读入 BGR，下面的 QImage 使用 RGB。
    cv::Mat rgb;
    cv::cvtColor(frame.image, rgb, cv::COLOR_BGR2RGB);

    QImage view(rgb.data, rgb.cols, rgb.rows,
        static_cast<qsizetype>(rgb.step),
        QImage::Format_RGB888);

    // 左侧：显示未经画框的原图。
    originalLabel_->setPixmap(
        QPixmap::fromImage(view.copy()).scaled(
            400, 650, Qt::KeepAspectRatio,
            Qt::SmoothTransformation));

    // 检测器只分析原图像素，返回候选框。
    //const InspectionResult result = inspectTraditional(frame.image);
    InspectionResult result;

    if (modeCombo_->currentIndex() == 0) {
        result = inspectTraditional(frame.image);
    }
    else {
        // 第一次选择 YOLO 时加载模型，以后点击“下一张”复用它。
        if (!yoloInspector_) {
            try {
                yoloInspector_ = std::make_unique<YoloInspector>(
                    std::filesystem::path(
                        LR"(T:\MyCode_qt\study\models\NEU-DET-YOLOv8n\neu_det_yolov8n_best.onnx)"
                    )
                );
            }
            catch (const std::exception& e) {
                QMessageBox::warning(
                    this, "模型加载失败", QString::fromUtf8(e.what()));
                return;
            }
        }

        result = yoloInspector_->inspect(frame.image);
    }

    if (!result.success) {
        imageLabel_->setText("检测失败");
        QMessageBox::warning(
            this, "检测失败",
            QString::fromStdString(result.errorMessage));
        return;
    }

    // 右侧：在原图副本上绘制每条检测结果的框。
    cv::Mat marked = frame.image.clone();

    for (const auto& detection : result.detections) {
        cv::rectangle(marked, detection.box, cv::Scalar(0, 0, 255), 2);
    }

    cv::Mat markedRgb;
    cv::cvtColor(marked, markedRgb, cv::COLOR_BGR2RGB);

    QImage markedView(markedRgb.data, markedRgb.cols, markedRgb.rows,
        static_cast<qsizetype>(markedRgb.step),
        QImage::Format_RGB888);

    imageLabel_->setPixmap(
        QPixmap::fromImage(markedView.copy()).scaled(
            400, 650, Qt::KeepAspectRatio,
            Qt::SmoothTransformation));

    fileLabel_->setText(
        QString::fromStdWString(frame.path.filename().wstring())
        + QString(" | 候选框：%1 | 耗时：%2 ms")
        .arg(static_cast<qulonglong>(result.detections.size()))
        .arg(result.elapsedMs, 0, 'f', 2));
}

void MainWindow::runBatch()
{
    if (currentFolder_.empty()) {
        QMessageBox::information(this, "批量检测", "请先选择图片文件夹");
        return;
    }
    if (batchThread_.isRunning()) {
        return;  // 一个时间只运行一个批次
    }

    // 启动时把参数固定下来；运行中改变界面选项不会改变本批任务。
    const QString folder =
        QString::fromStdWString(currentFolder_.wstring());
    const int mode = modeCombo_->currentIndex();
    const QString modelPath =
        QString::fromStdWString(
            LR"(T:\MyCode_qt\study\models\NEU-DET-YOLOv8n\neu_det_yolov8n_best.onnx)"
        );
    const QString reportPath =
        QString::fromStdWString(
            LR"(T:\MyCode_qt\study\IndustrialVision\out\batch_results.csv)"
        );

    auto* worker = new BatchWorker; // 不设置父对象，才能 moveToThread
    batchWorker_ = worker;
    worker->moveToThread(&batchThread_);

    // 线程启动后，process 在 Worker 所属的线程执行。
    connect(&batchThread_, &QThread::started, worker,
        [worker, folder, mode, modelPath, reportPath]() {
            worker->process(folder, mode, modelPath, reportPath);
        });

    // Worker 发出进度；这个接收函数在界面线程执行。
    connect(worker, &BatchWorker::progress, this,
        [this](int processed, int failed, int detected) {
            fileLabel_->setText(
                QString("已处理：%1 | 失败：%2 | 有检测结果：%3")
                .arg(processed).arg(failed).arg(detected));
        });

    connect(worker, &BatchWorker::finished, this,
        [this](int processed, int failed, int detected,
            qint64 totalMs, bool stopped, const QString& error) {
                if (!error.isEmpty()) {
                    fileLabel_->setText("批量失败：" + error);
                }
                else {
                    fileLabel_->setText(
                        QString("%1 | 处理：%2 | 失败：%3 | 有检测结果：%4 | %5 ms")
                        .arg(stopped ? "已停止" : "批量完成")
                        .arg(processed).arg(failed)
                        .arg(detected).arg(totalMs));
                }

                batchThread_.quit();
        });

    // 线程真正结束后，在工作线程中安排删除 Worker。
    connect(&batchThread_, &QThread::finished,
        worker, &QObject::deleteLater);

    batchButton_->setEnabled(false);
    stopButton_->setEnabled(true);
    fileLabel_->setText("批量检测中...");
    batchThread_.start();
}

//void MainWindow::runBatch()
//{
//    if (currentFolder_.empty()) {
//        QMessageBox::information(this, "批量检测", "请先选择图片文件夹");
//        return;
//    }
//
//    // 独立数据源：不影响界面“下一张”的当前位置。
//    ImageSequenceSource batchSource;
//    if (!batchSource.open(currentFolder_)) {
//        QMessageBox::warning(this, "批量检测", "无法打开图片文件夹");
//        return;
//    }
//
//    const int mode = modeCombo_->currentIndex();
//
//    // YOLO 模型只加载一次，在整个批次中复用。
//    if (mode == 1 && !yoloInspector_) {
//        try {
//            yoloInspector_ = std::make_unique<YoloInspector>(
//                std::filesystem::path(
//                    LR"(T:\MyCode_qt\study\models\NEU-DET-YOLOv8n\neu_det_yolov8n_best.onnx)"
//                )
//            );
//        }
//        catch (const std::exception& e) {
//            QMessageBox::warning(
//                this, "模型加载失败", QString::fromUtf8(e.what()));
//            return;
//        }
//    }
//
//    const std::filesystem::path reportPath =
//        R"(T:\MyCode_qt\study\IndustrialVision\out\batch_results.csv)";
//
//    std::ofstream report(reportPath);
//    if (!report) {
//        QMessageBox::warning(this, "批量检测", "无法创建 CSV 报告");
//        return;
//    }
//
//    report << "filename,success,detection_count,inference_ms\n";
//
//    int processed = 0;
//    int failed = 0;
//    int imagesWithDetections = 0;
//
//    QElapsedTimer timer;
//    timer.start();
//
//    while (true) {
//        ImageFrame frame;
//        std::string readError;
//
//        if (!batchSource.next(frame, readError)) {
//            if (readError.empty()) {
//                break;  // 正常到达文件夹末尾
//            }
//
//            ++failed;    // 坏图已经由数据源跳过，继续处理下一张
//            continue;
//        }
//
//        InspectionResult result =
//            (mode == 0)
//            ? inspectTraditional(frame.image)
//            : yoloInspector_->inspect(frame.image);
//
//        ++processed;
//
//        if (!result.success) {
//            ++failed;
//        }
//        else if (!result.detections.empty()) {
//            ++imagesWithDetections;
//        }
//
//        report << frame.path.filename().string() << ','
//            << (result.success ? 1 : 0) << ','
//            << result.detections.size() << ','
//            << result.elapsedMs << '\n';
//    }
//
//    report.close();
//
//    fileLabel_->setText(
//        QString("批量完成 | 处理：%1 | 失败：%2 | 有检测结果：%3 | 总耗时：%4 ms")
//        .arg(processed)
//        .arg(failed)
//        .arg(imagesWithDetections)
//        .arg(timer.elapsed())
//    );
//}