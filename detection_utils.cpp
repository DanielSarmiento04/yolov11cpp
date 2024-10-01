#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;



// Compute Intersection over Union (IoU) between two bounding boxes
float computeIOU(const Rect& box1, const Rect& box2) {
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    int interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    return (float)interArea / (box1Area + box2Area - interArea);
}


// Apply Soft-NMS (Soft Non-Maximum Suppression)
void applySoftNMS(std::vector<int>& indices, std::vector<float>& confidences, std::vector<Rect>& boxes, float sigma) {
    // Use Gaussian function to decay confidence scores
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx1 = indices[i];
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            float iou = computeIOU(boxes[idx1], boxes[idx2]);
            confidences[idx2] *= exp(-iou * iou / sigma);
            if (confidences[idx2] < 0.01f) {
                indices.erase(indices.begin() + j);
                --j;
            }
        }
    }
}


// Filter detections using Non-Maximum Suppression (NMS)
void filterDetections(const Mat& input, std::vector<int>& indices, std::vector<float>& confidences, std::vector<Rect>& boxes, float confThreshold, float nmsThreshold) {
    // Filter by confidence threshold
    for (size_t i = 0; i < confidences.size(); ++i) {
        if (confidences[i] >= confThreshold) {
            indices.push_back((int)i);
        }
    }

    // Apply Soft-NMS for improved bounding box filtering
    applySoftNMS(indices, confidences, boxes, 0.5f);  // Adjust sigma based on application
}



