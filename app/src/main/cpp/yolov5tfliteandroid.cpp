#include <jni.h>
#include <string>
#include "zlib.h"

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_yolov5tfliteandroid_MainActivity_extractTarGz(JNIEnv *env, jobject obj, jstring filePath) {
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
