#include "inference.h"
#include <algorithm>
#include <iostream>
#include <cmath> // For exp function

const std::vector<std::string> InferenceEngine::CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};


/**
 * @brief Letterbox an image to fit into the target size without changing its aspect ratio.
 * Adds padding to the shorter side to match the target dimensions.
 *
 * @param src Image to be letterboxed.
 * @param target_size Desired output size (width and height should be the same).
 * @param color Color of the padding (default is black).
 * @return Letterboxed image with padding.
 */
cv::Mat letterbox(const cv::Mat &src, const cv::Size &target_size, const cv::Scalar &color = cv::Scalar(0, 0, 0))
{
    // Calculate scale and padding
    float scale = std::min(target_size.width / (float)src.cols, target_size.height / (float)src.rows);
    int new_width = static_cast<int>(src.cols * scale);
    int new_height = static_cast<int>(src.rows * scale);

    // Resize the image with the computed scale
    cv::Mat resized_image;
    cv::resize(src, resized_image, cv::Size(new_width, new_height));

    // Create the output image with the target size and fill it with the padding color
    cv::Mat dst = cv::Mat::zeros(target_size.height, target_size.width, src.type());
    dst.setTo(color);

    // Calculate the top-left corner where the resized image will be placed
    int top = (target_size.height - new_height) / 2;
    int left = (target_size.width - new_width) / 2;

    // Place the resized image onto the center of the letterboxed image
    resized_image.copyTo(dst(cv::Rect(left, top, resized_image.cols, resized_image.rows)));

    return dst;
}

/**
 * @brief Computes the Intersection over Union (IoU) between two bounding boxes.
 *
 * @param boxA First bounding box.
 * @param boxB Second bounding box.
 * @return IoU value between 0 and 1.
 */
float computeIOU(const cv::Rect &boxA, const cv::Rect &boxB)
{
    int xA = std::max(boxA.x, boxB.x);
    int yA = std::max(boxA.y, boxB.y);
    int xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);

    int boxAArea = boxA.width * boxA.height;
    int boxBArea = boxB.width * boxB.height;

    float iou = static_cast<float>(interArea) / (boxAArea + boxBArea - interArea);
    return iou;
}


/**
 * @brief Applies Soft-NMS to a set of detected bounding boxes to reduce overlapping detections.
 *
 * @param detections Vector of detections to process.
 * @param sigma Soft-NMS parameter controlling the Gaussian function's width. Default is 0.5.
 * @param iou_threshold IoU threshold for suppression. Default is 0.3.
 */
void applySoftNMS(std::vector<Detection> &detections, float sigma = 0.5, float iou_threshold = 0.3)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            float iou = computeIOU(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold)
            {
                // Apply the Soft-NMS score decay formula
                detections[j].confidence *= std::exp(-iou * iou / sigma);
            }
        }
    }

    // Remove detections with low confidence scores
    detections.erase(std::remove_if(detections.begin(), detections.end(),
                                    [](const Detection &det) { return det.confidence < 0.001; }),
                     detections.end());
}


/**
 * @brief Apply Histogram Equalization to an image.
 *
 * @param src Input image in BGR format.
 * @return Image with enhanced contrast.
 */
cv::Mat applyHistogramEqualization(const cv::Mat &src)
{
    cv::Mat ycrcb_image;
    cv::cvtColor(src, ycrcb_image, cv::COLOR_BGR2YCrCb);  // Convert to YCrCb color space

    std::vector<cv::Mat> channels;
    cv::split(ycrcb_image, channels);

    // Apply histogram equalization to the Y channel (intensity)
    cv::equalizeHist(channels[0], channels[0]);

    // Merge back the channels and convert to BGR
    cv::merge(channels, ycrcb_image);
    cv::Mat result;
    cv::cvtColor(ycrcb_image, result, cv::COLOR_YCrCb2BGR);

    return result;
}

/**
 * @brief Apply CLAHE to an image for adaptive contrast enhancement.
 *
 * @param src Input image in BGR format.
 * @return Image with enhanced local contrast.
 */
cv::Mat applyCLAHE(const cv::Mat &src)
{
    cv::Mat lab_image;
    cv::cvtColor(src, lab_image, cv::COLOR_BGR2Lab);  // Convert to LAB color space

    std::vector<cv::Mat> lab_planes;
    cv::split(lab_image, lab_planes);

    // Apply CLAHE to the L channel (lightness)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);  // Set the clip limit for contrast enhancement
    clahe->apply(lab_planes[0], lab_planes[0]);

    // Merge the planes back and convert to BGR
    cv::merge(lab_planes, lab_image);
    cv::Mat result;
    cv::cvtColor(lab_image, result, cv::COLOR_Lab2BGR);

    return result;
}


/**
 * @brief Apply Gamma Correction to an image.
 *
 * @param src Input image in BGR format.
 * @param gamma Gamma value for correction. Values < 1 will lighten the image, values > 1 will darken it.
 * @return Image with gamma correction applied.
 */
cv::Mat applyGammaCorrection(const cv::Mat &src, float gamma)
{
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i)
    {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }

    cv::Mat result;
    cv::LUT(src, lut, result);  // Apply the gamma lookup table to the image

    return result;
}


InferenceEngine::InferenceEngine(const std::string &model_path)
{
   
   
}

InferenceEngine::~InferenceEngine() {}
