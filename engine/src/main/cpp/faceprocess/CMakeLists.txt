cmake_minimum_required(VERSION 3.19)

project(FaceProcessingTest LANGUAGES C CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
# message(OpenCV "====================${OpenCV_LIBS}")

include_directories(
  /tmp/ncnn/build/install/include
  # /tmp/tensorflow-2.12.0/
)
link_directories(
  /tmp/ncnn/build/install/lib
  # /tmp/tensorflow-2.12.0/bazel-bin/tensorflow/lite/c/
)

add_library(facedetector STATIC
  face_detector.cpp
)
target_link_libraries(facedetector PRIVATE
  ncnn
  ${OpenCV_LIBS}
)

add_executable(test
  test.cpp)
target_link_libraries(test
  facedetector
  ${OpenCV_LIBS}
)