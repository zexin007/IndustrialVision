#include <opencv2/opencv.hpp>
#include "industrial_vision/AnnotationPreview.h"
#include "industrial_vision/MaskUtils.h"


bool createGroundTruthPreview(
    const std::string& imagePath,
    const std::string& maskPath,
    const std::string& outputPath,
    std::string& errorMessage)
{
    errorMessage.clear();

    try {
        const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            errorMessage = "Could not read source image: " + imagePath;
            return false;
        }

        const cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            errorMessage = "Could not read ground-truth mask: " + maskPath;
            return false;
        }

        if (image.size() != mask.size()) {
            errorMessage = "Image and mask sizes do not match.";
            return false;
        }

        const auto defectBoxes =
            industrial_vision::extractDefectBoxes(mask);

        cv::Mat result = image.clone();

        for (const auto& box : defectBoxes) {
            cv::rectangle(result, box, cv::Scalar(0, 0, 255), 2);
        }

        if (!cv::imwrite(outputPath, result)) {
            errorMessage = "Could not save result image: " + outputPath;
            return false;
        }

        return true;
    }
    catch (const cv::Exception& e) {
        errorMessage = e.what();
        return false;
    }
}