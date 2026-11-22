// IndustrialVision.cpp: 定义应用程序的入口点。
//

#include "industrial_vision/MaskUtils.h"
#include "industrial_vision/AnnotationPreview.h"
#include "industrial_vision/ImageSequenceSource.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

int main() {

	const std::filesystem::path trainDir = R"(T:\简历\find a job\KolektorSDD2\full_dataset\train)";
	const std::filesystem::path outputDir = R"(T:\MyCode_qt\study\IndustrialVision\out\result)";

	// 放在 main() 开头
	ImageSequenceSource source;
	if (!source.open(trainDir)) {
		std::cerr << "Failed to open image folder\n";
		return 1;
	}
	ImageFrame frame;
	std::string errorMessage;

	for (int i = 0; i < 10; ++i) {
		if (!source.next(frame, errorMessage)) {
			if (!errorMessage.empty()) {
				std::cerr << errorMessage << '\n';  // 坏图
			}
			break;                                  // 或正常读完
		}

		std::cout << frame.id << ": "
			<< frame.path.filename().string() << "  "
			<< frame.image.cols << "x" << frame.image.rows << '\n';
	}

	std::filesystem::create_directories(outputDir);

	int processed = 0;
	int successCount = 0;
	int failedCount = 0;

	for (const auto& entry : std::filesystem::directory_iterator(trainDir)) {
		std::string filename = entry.path().filename().string();
		// 只处理文件名以 _GT.png 结尾的掩码
		if (filename.size() < 7 ||
			filename.substr(filename.size() - 7) != "_GT.png") {
			continue;
		}

		const std::string imageName =filename.substr(0, filename.size() - 7) + ".png";
		
		// 拼接图像路径，/是std::filesystem::path的操作符重载，可以直接拼接路径
		const std::filesystem::path imagePath = trainDir / imageName;
		const std::filesystem::path maskPath = entry.path();
		const std::filesystem::path outputPath = outputDir / (filename.substr(0, filename.size() - 7) + "_boxes.png");

		++processed;
		std::string errorMessage;
		if (createGroundTruthPreview(imagePath.string(), maskPath.string(), outputPath.string(), errorMessage)) {
			++successCount;
			std::cout << "Saved: " << outputPath << '\n';
		}
		else {
			++failedCount;
			std::cerr << "Failed to process: " << imageName << ". Error: " << errorMessage << '\n';
		}
		if (processed == 20) {
			break;
		}

	}
	std::cout << "Processed: " << processed
		<< ", success: " << successCount
		<< ", failed: " << failedCount << '\n';
	return failedCount == 0 ? 0 : 1;

}