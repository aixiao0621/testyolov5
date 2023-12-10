#ifndef FACE_DETECTION_HPP
#define FACE_DETECTION_HPP

#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

struct FaceInfo{
    float left;
    float top;
    float right;
    float bottom;
    float score;
    float label;
};


class FaceDetector{
private:

    // threshold
    const float confThreshold = 0.0;
    const float nmsThreshold = 0.0;

    std::shared_ptr<char>mModelBuffer{};
    long mModelSize;

    std::unique_ptr<tflite::FlatBufferModel> mModel{};
    std::unique_ptr<tflite::Interpreter> mInterpreter{};

    int _input;
    int _in_height;
    int _in_width;
    int _in_channels;
    int _in_type;

    int _img_height;
    int _img_width;

    uint8_t *_input_8;

    const size_t nthreads = 1;

public:
    FaceDetector(char* buffer, long size);

    FaceDetector(const FaceDetector&) = delete;

    FaceDetector& operator=(const FaceDetector&) = delete;

    FaceDetector(FaceDetector&&) = delete;

    FaceDetector& operator=(FaceDetector&&) = delete;

    ~FaceDetector();

    void detect(
        cv::Mat& img,
        std::vector<FaceInfo>& faces);

    static void array2Mat(char* bytes, cv::Mat& mat, int h, int w);

    void preProcess(cv::Mat& img);

    void tensor2Vector2d(
            const TfLiteTensor* tensor,
            std::vector<std::vector<float>>& predV,
            const int row,
            const int col);

private:
    void loadModel();

    void fill(uint8_t *in, cv::Mat &src);

    void nonMaximumSupprition(std::vector<std::vector<float>>& predV,
                              std::vector<cv::Rect> &boxes,
                              std::vector<float> &confidences,
                              std::vector<int> &classIds,
                              std::vector<int> &indices,
                              const int &row,
                              const int &colum);

};

#endif
