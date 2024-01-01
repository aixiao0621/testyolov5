#include <jni.h>
#include <string>

#include "zlib.h"

#include "face_detection.hpp"


char* ConvertJByteaArrayToChars(JNIEnv *env, jbyteArray bytearray){
    char *chars = nullptr;
    jbyte *bytes;

    bytes = env->GetByteArrayElements(bytearray, 0);
    int chars_len = env->GetArrayLength(bytearray);
    chars = new char[chars_len];
    memcpy(chars, bytes, chars_len);
    chars[chars_len] = 0;

    env->ReleaseByteArrayElements(bytearray, bytes, 0);

    return chars;
}


extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_yolov5tfliteandroid_JNITools_extractTarGz(JNIEnv *env, jobject obj, jstring filePath) {
    const char *file_path = env->GetStringUTFChars(filePath, nullptr);
    gzFile file = gzopen(file_path, "rb");
    if (file == nullptr) {
        // 处理文件打开失败的情况
        env->ReleaseStringUTFChars(filePath, file_path);
        return env->NewByteArray(2);
    }

    const int BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    std::string extractedData;

    int uncompressed_bytes;
    while ((uncompressed_bytes = gzread(file, buffer, sizeof(buffer))) > 0) {
        extractedData.append(buffer, uncompressed_bytes);
    }

    gzclose(file);

    env->ReleaseStringUTFChars(filePath, file_path);

    // 将提取的数据转换为Java的byte数组
    jbyteArray resultData = env->NewByteArray(extractedData.size());
    env->SetByteArrayRegion(resultData, 0, extractedData.size(), reinterpret_cast<const jbyte *>(extractedData.c_str()));
    return resultData;
}

JNICALL extern "C"
JNIEXPORT jlong Java_com_example_yolov5tfliteandroid_JNITools_createDetector(
        JNIEnv *env,
        jobject obj,
        jbyteArray modelBuffer,
        jlong modelSize) {

    jbyte *buffer = env->GetByteArrayElements(modelBuffer, 0);
    void* detector = new FaceDetector(reinterpret_cast<char *>(buffer), modelSize);
    env->ReleaseByteArrayElements(modelBuffer, buffer, 0);
    return (jlong) detector;
}

JNICALL extern "C"
JNIEXPORT jfloatArray Java_com_example_yolov5tfliteandroid_JNITools_detect(
        JNIEnv *env,
        jobject obj,
        jlong detectorPtr,
        jbyteArray src,
        jint width,
        jint height) {
    const size_t kFaceAttb = 6;
    jfloatArray detections;
    auto detector = reinterpret_cast<FaceDetector *>(detectorPtr);
    cv::Mat desMat{};
    std::vector<FaceInfo> res;

    // convert src to char[]
    char* src_data = reinterpret_cast<char *>(env->GetByteArrayElements(src, 0));

    FaceDetector::array2Mat(src_data, desMat, height, width);
    detector->detect(desMat, res);
    const auto arr_len = kFaceAttb * res.size();
    if (arr_len == 0) {
        return env->NewFloatArray(0);
    }
    detections = env->NewFloatArray(arr_len);

    env->SetFloatArrayRegion(detections, 0, arr_len, reinterpret_cast<const jfloat *>(res.data()));

    return detections;
}