#pragma once
#include <string>

bool createGroundTruthPreview(
    const std::string& imagePath,
    const std::string& maskPath,
    const std::string& outputPath,
    std::string& errorMessage);