#pragma once

#include <filesystem>
#include <vector>
#include <opencv2/core.hpp>

#include <chrono>
#include <string>

struct ImageFrame {
    std::size_t id = 0;
    std::filesystem::path path;
    cv::Mat image;
    std::chrono::steady_clock::time_point acquiredAt{};
};
class ImageSequenceSource {
public:
    bool open(const std::filesystem::path& folder);
    const std::vector<std::filesystem::path>& files() const;
	//bool next(std::filesystem::path& file);
    bool next(ImageFrame& frame, std::string& errorMessage);
   
private:
    std::vector<std::filesystem::path> files_;
    std::size_t currentIndex_ = 0;
};