<h1 align="center">Yolo V11 cpp</h1>

<h3 align="center"> Jose Sarmiento | josedanielsarmiento219@gmail.com</h3>


## Resume

This repository provides a C++ implementation to run the **YOLOv11** object detection model using **OpenCV** and **ONNX**. The project supports running the YOLOv11 model in real-time on images, videos, or camera streams by leveraging OpenCV's DNN module for ONNX inference or using the ONNX Runtime C++ API for optimized execution.

## Features
- **Real-time object detection** using YOLOv11 in C++.
- Supports **image**, **video**, and **webcam** inference.
- Two modes of inference:
  - **OpenCV DNN module** for lightweight execution.
  - **ONNX Runtime C++ API** for optimized performance on different hardware (CPU, GPU, etc.).
- Customizable confidence threshold and non-max suppression (NMS).

## Requirements

### 1. Dependencies
To build and run the project, you need the following dependencies:

- OpenCV 4.x with DNN support.
- ONNX Runtime C++ (optional for ONNX Runtime inference).
- CMake (for building the project).

Install OpenCV:

```bash
    sudo apt-get install libopencv-dev
```

Install ONNX Runtime:

```
    git clone https://github.com/microsoft/onnxruntime
    cd onnxruntime
    ./build.sh --config Release --build_shared_lib --parallel
```

## Usage

1. Build the Project
Clone the repository and build using CMake:

```
    git clone https://github.com/DanielSarmiento04/yolov11cpp
    cd yolov11cpp
    mkdir build && cd build
    cmake ..
    make
```