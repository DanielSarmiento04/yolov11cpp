#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#define MAX_STRIDE 32

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

std::string getInputName(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

std::string getOutputName(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetOutputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

float clamp(float val, float min = 0.f, float max = 1280.f) {
    return std::max(min, std::min(max, val));
}

void non_max_suppression(
    std::vector<Object>& proposals, std::vector<Object>& results,
    int orin_h, int orin_w, float conf_thres = 0.25f, float iou_thres = 0.65f
) {
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> indices;

    for (const auto& obj : proposals) {
        bboxes.push_back(obj.rect);
        scores.push_back(obj.prob);
    }

    cv::dnn::NMSBoxes(bboxes, scores, conf_thres, iou_thres, indices);

    for (int i : indices) {
        results.push_back(proposals[i]);
    }
}

void preprocess(const cv::Mat& img, cv::Mat& out, int target_size = 640) {
    int img_w = img.cols;
    int img_h = img.rows;

    float scale = std::min(target_size / (float)img_w, target_size / (float)img_h);
    int new_w = img_w * scale;
    int new_h = img_h * scale;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;

    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    out.convertTo(out, CV_32F, 1.0 / 255);
}

void postprocess(const float* output, int rows, int cols, float conf_threshold, std::vector<Object>& objects) {
    for (int i = 0; i < rows; ++i) {
        float confidence = output[i * cols + 4];  // Object confidence

        if (confidence >= conf_threshold) {
            float x = output[i * cols];
            float y = output[i * cols + 1];
            float w = output[i * cols + 2];
            float h = output[i * cols + 3];
            int label = std::max_element(output + i * cols + 5, output + (i + 1) * cols) - (output + i * cols + 5);
            float prob = confidence;

            Object obj;
            obj.rect = cv::Rect_<float>(x - w / 2, y - h / 2, w, h);
            obj.label = label;
            obj.prob = prob;
            objects.push_back(obj);
        }
    }
}

void detect_yolov11(const cv::Mat& img, Ort::Session& session, std::vector<Object>& objects) {
    cv::Mat input;
    preprocess(img, input);

    std::vector<int64_t> input_shape = {1, 3, input.rows, input.cols};
    std::vector<float> input_tensor_values(input.begin<float>(), input.end<float>());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_name = getInputName(session).c_str();
    const char* output_name = getOutputName(session).c_str();

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int rows = output_shape[1];
    int cols = output_shape[2];

    postprocess(output_data, rows, cols, 0.25f, objects);
}

void draw_objects(const cv::Mat& img, const std::vector<Object>& objects) {
    cv::Mat image = img.clone();

    for (const auto& obj : objects) {
        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0), 2);

        std::string label = std::to_string(obj.label) + ": " + std::to_string(int(obj.prob * 100)) + "%";
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int x = std::max(0, std::min((int)obj.rect.x, image.cols - label_size.width));
        int y = std::max(label_size.height, (int)obj.rect.y);

        cv::rectangle(image, cv::Rect(cv::Point(x, y - label_size.height), label_size + cv::Size(0, baseline)),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite("output.jpg", image);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <imagepath>" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov11(img, session, objects);
    draw_objects(img, objects);

    return 0;
}
