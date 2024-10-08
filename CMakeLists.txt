cmake_minimum_required(VERSION 3.10)

# Set the project name in a variable
set(project_name yolov11_cpp)
project(${project_name})
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Find ONNX Runtime package
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    HINTS /opt/homebrew/Cellar/onnxruntime/1.17.1/include/onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS /opt/homebrew/Cellar/onnxruntime/1.17.1/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime include directory not found")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()

add_library(${project_name}-lib
    detection_utils.cpp
    image_utils.cpp

)

target_include_directories(${project_name}-lib PUBLIC src)
target_include_directories(${project_name}-lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})

target_link_libraries(${project_name}-lib
    PUBLIC ${OpenCV_LIBS}
    PUBLIC ${ONNXRUNTIME_LIBRARY}
)

# Add the main executable
add_executable(${project_name} 
    ./main.cpp
)
target_include_directories(${project_name} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(${project_name} ${project_name}-lib)
