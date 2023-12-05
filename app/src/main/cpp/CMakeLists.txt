# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

cmake_minimum_required(VERSION 3.22.1)

project("yolov5tfliteandroid")

add_library(${CMAKE_PROJECT_NAME} SHARED
    yolov5tfliteandroid.cpp
    face_detection.cpp
    face_detection.hpp
)
set(INCLUDE_DIRS
        ${OpenCV_DIR}/jni/include
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})
add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
        ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so)

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/${ANDROID_ABI}/libtensorflowlite.so)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
    # List libraries link to the target library
    android
    log)
# 设置_FORTIFY_SOURCE宏定义
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE -D_FORTIFY_SOURCE=2)

# 导入zlib库
find_library(zlib-lib z)
target_link_libraries(yolov5tfliteandroid ${zlib-lib} lib_opencv tflite)
