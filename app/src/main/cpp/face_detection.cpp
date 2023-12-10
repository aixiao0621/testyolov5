#include "face_detection.hpp"

FaceDetector::FaceDetector(char* buffer, long size){
    mModelSize = size;
    mModelBuffer.reset(new char[mModelSize]);
    memcpy(mModelBuffer.get(), buffer, sizeof(char) * mModelSize);
    loadModel();
}

void FaceDetector::loadModel(){

    mModel = tflite::FlatBufferModel::BuildFromBuffer(
            mModelBuffer.get(),
            sizeof(char) * mModelSize);

    assert(mModel != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*mModel, resolver);
    builder(&mInterpreter);
    assert(mInterpreter != nullptr);

    TfLiteStatus status = mInterpreter->AllocateTensors();

    // input information
    _input = mInterpreter->inputs()[0];
    TfLiteIntArray *dims = mInterpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _in_channels = dims->data[3];
    _in_type = mInterpreter->tensor(_input)->type;
    _input_8 = mInterpreter->typed_tensor<uint8_t>(_input);
    _in_type = mInterpreter->tensor(_input)->type;

    mInterpreter->SetNumThreads(nthreads);


    assert(status == kTfLiteOk);

    assert(mInterpreter->inputs().size() == 1);

}

void FaceDetector:: array2Mat(char* bytes, cv::Mat& mat, int h, int w) {
    // 检查输入参数
    assert(bytes != nullptr);
    assert(mat.empty());
    assert(h > 0);
    assert(w > 0);

    // 初始化输出图像
    mat.create(h, w, CV_8UC3);

    // 遍历图像像素
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            // 从 BGRA 转换为 RGB
            mat.at<cv::Vec3b>(i, j)[0] = bytes[i * w * 4 + j * 4];
            mat.at<cv::Vec3b>(i, j)[1] = bytes[i * w * 4 + j * 4 + 1];
            mat.at<cv::Vec3b>(i, j)[2] = bytes[i * w * 4 + j * 4 + 2];
        }
    }
}

void FaceDetector::nonMaximumSupprition(std::vector<std::vector<float>>& predV,
                                        std::vector<cv::Rect> &boxes,
                                        std::vector<float> &confidences,
                                        std::vector<int> &classIds,
                                        std::vector<int> &indices,
                                        const int &row,
                                        const int &colum) {
    std::vector<cv::Rect> boxesNMS;
    std::vector<float> scores;
    double confidence;
    cv::Point classId;

    for (int i = 0; i < row; i++){
        if (predV[i][4] > confThreshold){
            // height--> image.rows,  width--> image.cols;
            int left = (predV[i][0] - predV[i][2] / 2) * _img_width;
            int top = (predV[i][1] - predV[i][3] / 2) * _img_height;
            int w = predV[i][2] * _img_width;
            int h = predV[i][3] * _img_height;

            for (int j = 5; j < colum; j++)
            {
                // conf = obj_conf * cls_conf
                scores.push_back(predV[i][j] * predV[i][4]);
            }

            cv::minMaxLoc(
                    scores,
                    0,
                    &confidence,
                    0,
                    &classId);

            if (confidence > confThreshold) {
                boxes.push_back(cv::Rect(left, top, w, h));
                confidences.push_back(confidence);
                classIds.push_back(classId.x);
                boxesNMS.push_back(cv::Rect(left, top, w, h));
            }
        }
    }

    cv::dnn::NMSBoxes(
            boxesNMS,
            confidences,
            confThreshold,
            nmsThreshold,
            indices);
}

void FaceDetector::preProcess(cv::Mat &img) {
    cv::resize(
            img,
            img,
            cv::Size(
                    _in_width,
                    _in_height),
            cv::INTER_CUBIC);
//    img.convertTo(img, CV_8U);
}

void FaceDetector::detect(
    cv::Mat &img,
    std::vector<FaceInfo> &res) {

    _img_height = img.rows;
    _img_width = img.cols;

    preProcess(img);
    fill(_input_8, img);

    auto state = mInterpreter->Invoke();
    if (state!= kTfLiteOk) {
        throw std::runtime_error("Invoke failed");
    }

    int _out = mInterpreter->outputs()[0];
    TfLiteIntArray *_out_dims = mInterpreter->tensor(_out)->dims;
    int _out_row   = _out_dims->data[1];   // 25200
    int _out_colum = _out_dims->data[2];   // class number + 5 ---> 85     bbox cond class

    TfLiteTensor *pOutputTensor = mInterpreter->tensor(mInterpreter->outputs()[0]);

    std::vector<std::vector<float>> predV{};
    tensor2Vector2d(pOutputTensor, predV, _out_row, _out_colum);

    std::vector<int> indices;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    nonMaximumSupprition(predV,
                         boxes,
                         confidences,
                         classIds,
                         indices,
                         _out_row,
                         _out_colum);

    for (int i = 0; i < indices.size(); i++) {
        res.push_back(FaceInfo{
                static_cast<float>(boxes[indices[i]].x),
                static_cast<float>(boxes[indices[i]].y),
                static_cast<float>(boxes[indices[i]].x + boxes[indices[i]].width),
                static_cast<float>(boxes[indices[i]].y + boxes[indices[i]].height),
                confidences[indices[i]],
                static_cast<float>(classIds[indices[i]])
            });
    }
}

void FaceDetector::tensor2Vector2d(
        const TfLiteTensor *tensor,
        std::vector<std::vector<float>> &predV,
        const int row,
        const int col) {
    auto scale = tensor->params.scale;
    auto zero_point = tensor->params.zero_point;
    for (int32_t i = 0; i < row; i++)
    {
        std::vector<float> _tem;
        for (int j = 0; j < col; j++)
        {
            float val_float = (((int32_t)tensor->data.uint8[i * col + j]) - zero_point) * scale;
            _tem.push_back(val_float);
        }
        predV.push_back(_tem);
    }
}
// 👇👇👇👇👇👇
void FaceDetector::fill(uint8_t *in, cv::Mat &src) {
    int n = 0, nc = src.channels(), ne = src.elemSize();

    if (src.isContinuous()){
        memcpy(in, src.data, nc * src.cols * src.rows);
        return;
    }
//
//    for (int y = 0; y < src.rows; ++y)
//        for (int x = 0; x < src.cols; ++x)
//            for (int c = 0; c < nc; ++c)
//                in[n++] = src.data[y * src.step + x * ne + c];
}

FaceDetector::~FaceDetector() = default;