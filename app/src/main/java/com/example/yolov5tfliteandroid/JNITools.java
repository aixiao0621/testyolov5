package com.example.yolov5tfliteandroid;

public class JNITools {
    static {
        System.loadLibrary("yolov5tfliteandroid");
    }
    public native byte[] extractTarGz(String filePath);
    public native long createDetector(byte[] modelBuffer, long size);
    public native float[] detect(long detectorPtr,
                                 byte[] src,
                                 int width,
                                 int height);
}
