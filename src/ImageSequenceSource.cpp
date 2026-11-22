#include "industrial_vision/ImageSequenceSource.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>

bool ImageSequenceSource::open(const std::filesystem::path& folder) {
    files_.clear();
    currentIndex_ = 0;

    if (folder.empty() ||
        !std::filesystem::exists(folder) ||
        !std::filesystem::is_directory(folder)) {
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto path = entry.path();
        const std::string filename = path.filename().string();

        const auto ext = path.extension();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") {
            continue;
        }

        if (filename.size() >= 7 &&
            filename.substr(filename.size() - 7) == "_GT.png") {
            continue;
        }

        files_.push_back(path);
    }

    std::sort(files_.begin(), files_.end());
    return !files_.empty();
}

bool ImageSequenceSource::next(ImageFrame& frame, std::string& errorMessage) {
    frame = {};
	errorMessage.clear();
    if (currentIndex_ >= files_.size()) {
        return false;
    }
	const auto& path = files_[currentIndex_++];

    frame.path = path;
    frame.id = currentIndex_;

	try { 
        frame.image = cv::imread(path.string(), cv::IMREAD_COLOR);
    }
	catch (const cv::Exception& e) {
		errorMessage = "Failed to read image: " + path.string() + ". Error: " + e.what();
		return false;
	}
    if (frame.image.empty()) {
        errorMessage = "Failed to read image: " + path.string();
        return false;
    }

    frame.id = currentIndex_;
    frame.path = path;
    return true;
}