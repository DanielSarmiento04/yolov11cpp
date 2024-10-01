#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath> 

struct Detection
{
    float confidence;
    cv::Rect bbox;
    int class_id;
    std::string class_name;
};


class InferenceEngine
{
public:
    InferenceEngine(const std::string &model_path);
    ~InferenceEngine();

   
    std::vector<int64_t> input_shape;
    
private:
    
    static const std::vector<std::string> CLASS_NAMES;
};


#endif // INFERENCE_H
