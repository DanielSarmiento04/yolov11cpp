#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Apply a letterbox to keep aspect ratio consistent
void applyLetterbox(const Mat& src, Mat& dst, int inputWidth, int inputHeight) {
    float scale = min((float)inputWidth / src.cols, (float)inputHeight / src.rows);
    int newWidth = (int)(src.cols * scale);
    int newHeight = (int)(src.rows * scale);
    int xOffset = (inputWidth - newWidth) / 2;
    int yOffset = (inputHeight - newHeight) / 2;

    resize(src, dst, Size(newWidth, newHeight));
    copyMakeBorder(dst, dst, yOffset, inputHeight - newHeight - yOffset, xOffset, inputWidth - newWidth - xOffset, BORDER_CONSTANT, Scalar(114, 114, 114));
}

// Preprocess the input image
Mat preprocessImage(const Mat& img, int inputWidth, int inputHeight) {
    Mat result;
    applyLetterbox(img, result, inputWidth, inputHeight);
    applyHistogramEqualization(result);  // Optional step for better contrast
    return result;
}

// Apply histogram equalization to enhance contrast
void applyHistogramEqualization(Mat& img) {
    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, ycrcb);
    cvtColor(ycrcb, img, COLOR_YCrCb2BGR);
}
