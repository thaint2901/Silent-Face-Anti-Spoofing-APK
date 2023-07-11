cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../../distribution/${ANDROID_ABI})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libncnn.a)

add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_core.so)

add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_imgproc.so)

add_library(opencv_codecs STATIC IMPORTED)
set_target_properties(opencv_codecs PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_imgcodecs.so)


file(GLOB jni_srcs
        "*.cpp"
    )

add_library(faceprocess STATIC
        faceprocess/mtcnn.cpp
        faceprocess/live.cpp)
target_link_libraries(faceprocess PRIVATE
        ncnn opencv_core opencv_imgproc opencv_codecs)

add_library(engine SHARED ${jni_srcs})

target_link_libraries(engine faceprocess
        log jnigraphics ncnn opencv_core opencv_imgproc opencv_codecs android)