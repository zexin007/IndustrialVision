# IndustrialVision

基于 C++17、OpenCV、ONNX Runtime 和 Qt Widgets 的工业视觉学习项目。

当前以钢带表面缺陷检测为模拟场景：用图片文件夹代替相机采图，分别尝试传统视觉和 YOLO 检测，并研究连续出帧时的处理节拍、线程协作和过载问题。项目尚未接入真实工业相机、PLC 或标记设备。

## 目前能做什么

- 从图片文件夹按顺序读取图像。
- 使用传统视觉方法或 ONNX 格式的 YOLO 模型检测图像。
- 在 Qt 窗口中对比原图和画框结果，手动切换到下一张。
- 在工作线程中批量检测，显示进度并支持停止，生成 CSV 报告。
- 使用 `LiveProbe` 模拟按节拍连续出帧，观察排队、检测延迟和过载。

传统视觉方法目前只是基线，检测效果尚不理想。少量图片上的 YOLO 可视化结果也不能代表真实产线检出率。

## 项目结构与文件作用

### 项目配置

| 文件                    | 作用                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `CMakeLists.txt`        | 定义编译目标、依赖关系及 Windows 运行时 DLL 的复制规则。     |
| `CMakePresets.json`     | 定义通用的 CMake 配置预设，例如编译器、架构和 Debug/Release。 |
| `CMakeUserPresets.json` | 保存本机使用的依赖路径等配置。它与具体电脑环境相关。         |

### 公共数据与图像来源

| 文件                                              | 作用                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `include/industrial_vision/ImageSequenceSource.h` | 声明 `ImageFrame` 和 `ImageSequenceSource`。`ImageFrame` 保存帧编号、路径、图像及模拟采集时间。 |
| `src/ImageSequenceSource.cpp`                     | 扫描、筛选和排序文件；`next()` 逐张读取并解码图像，区分读到末尾与读图失败。 |
| `include/industrial_vision/InspectionResult.h`    | 定义统一的 `Detection` 和 `InspectionResult`，供不同检测方法返回结果。 |

`ImageFrame::id` 在当前模拟中用作钢带段号。真实产线中的位置应由相机触发或编码器等现场信号确定；目前按段号计算的位置只是软件假设。

### 检测与标注工具

| 文件                                               | 作用                                                         |
| -------------------------------------------------- | ------------------------------------------------------------ |
| `include/industrial_vision/TraditionalInspector.h` | 声明传统视觉检测接口。                                       |
| `src/TraditionalInspector.cpp`                     | 使用灰度图、黑帽、阈值、形态学及连通域提取候选区域；不读取 GT。 |
| `include/industrial_vision/YoloInspector.h`        | 声明 YOLO 检测器；通过内部实现对象管理 ONNX Runtime 资源。   |
| `src/YoloInspector.cpp`                            | 加载并复用 ONNX 模型，完成预处理、推理、输出解码、NMS 和坐标还原。 |
| `include/industrial_vision/MaskUtils.h`            | 声明从二值掩码提取矩形框的工具函数。                         |
| `src/MaskUtils.cpp`                                | 对掩码查找轮廓，并把轮廓转换为外接矩形。                     |
| `include/industrial_vision/AnnotationPreview.h`    | 声明 GT 标注预览函数。                                       |
| `src/AnnotationPreview.cpp`                        | 读取原图和 GT 掩码，把由 GT 提取的框画到预览图并保存。       |

`MaskUtils` 和 `AnnotationPreview` 用于**查看数据集标注**。GT 不能参与在线检测，也不能把 GT 框当作算法预测框。

### Qt 界面与批处理

| 文件                     | 作用                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `src/QtMain.cpp`         | Qt 程序入口：创建 `QApplication` 和 `MainWindow`。           |
| `src/ui/MainWindow.h`    | 声明窗口控件、图片源、检测器及批处理线程等成员。             |
| `src/ui/MainWindow.cpp`  | 实现选择文件夹、切换检测方法、显示原图与检测结果、启动和停止批处理。 |
| `src/ui/BatchWorker.h`   | 声明批处理工作对象，以及进度、完成信号和停止标志。           |
| `src/ui/BatchWorker.cpp` | 在工作线程中逐张读取和检测，复用模型，写入 CSV，并在合适的位置响应停止请求。 |

当前 Qt 窗口中的“下一张”是手动触发的单张检测；批量检测在工作线程运行。`LiveProbe` 的连续出帧流程**尚未接入 Qt 界面**。

### 独立程序

| 文件                       | 作用                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `src/IndustrialVision.cpp` | 早期控制台程序，主要用于图片遍历和 GT 标注预览；不是当前 Qt 检测程序的入口。 |
| `src/ModelProbe.cpp`       | 模型调试程序：验证 ONNX 推理、输出格式、画框及与标签的可视化对比。 |
| `src/LiveProbe.cpp`        | 连续检测实验程序：采集线程按节拍读图，检测线程消费有限队列；统计处理耗时、队列峰值和过载段号。 |

这些独立程序用于验证不同问题，不应与 `IndustrialVisionQt` 混淆。

## CMake 目标之间的关系

```text
vision_core
├── 图片来源 ImageSequenceSource
├── 统一结果类型 InspectionResult
├── 传统视觉 TraditionalInspector
└── GT 预览工具 MaskUtils / AnnotationPreview

vision_inference
├── YoloInspector
└── 依赖 vision_core、ONNX Runtime

IndustrialVisionQt
├── QtMain / MainWindow / BatchWorker
└── 依赖 vision_core、vision_inference、Qt Widgets

IndustrialVision → 早期控制台程序，依赖 vision_core
ModelProbe       → 单图模型调试程序
LiveProbe        → 连续出帧与过载实验程序，依赖 vision_inference
```

编译时需要找到头文件和链接库；程序运行时还需要找到对应的 OpenCV、ONNX Runtime 和 Qt DLL。`CMakeLists.txt` 中的部分 `POST_BUILD` 命令负责复制运行所需的 DLL。

## 当前连续检测的数据流

```text
图片文件夹
    ↓
采集线程：按模拟节拍调用 ImageSequenceSource::next()
    ↓
有限队列：最多暂存 3 帧
    ↓
检测线程：YoloInspector::inspect()
    ↓
打印段号、候选框数量、处理耗时、模拟位置与是否超期
```

队列满时，`LiveProbe` 停止本次模拟并报告过载；尚未检测的段号必须标记为未检，不能当作 OK。这里的 `acquire-to-result` 从文件**读完之后**开始计时，尚不包含真实相机曝光、图像传输、PLC 通信和执行机构动作。

## 后续开发

1. 将连续检测流程整理为可由 Qt 启动、停止和监控的模块，显示最新画面及故障状态。
2. 建立模拟钢带位置与结果的对应关系，处理超期、未检和重复结果。
3. 接入本机控制端模拟器，定义消息、确认、超时及断线处理。
4. 补充配置、日志、缺陷证据保存、定量评估和可独立运行的发布包。
5. 同步学习相机与镜头选型、打光、触发、相机 SDK 缓冲区管理、PLC 协议和现场验收方法。

真实相机取像质量、编码器位置、控制设备动作和产线检测指标，需要现场设备与样本才能验证。本项目会分别记录软件仿真结果和仍需现场验证的事项。