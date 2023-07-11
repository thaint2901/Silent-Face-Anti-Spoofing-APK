//
// Created by yuanhao on 20-6-10.
//

#ifndef LIVEBODYEXAMPLE_FACE_DETECTOR_H
#define LIVEBODYEXAMPLE_FACE_DETECTOR_H

#include <opencv2/core/mat.hpp>
#include "ncnn/net.h"

#include "definition.h"


class FaceDetector {
public:
    FaceDetector();

    ~FaceDetector();

    void setMinFaceSize(int size);

#if __ANDROID_API__ >= 9
    int loadModel(AAssetManager* assetManager);
#else
    int loadModel(const char* emb_model_bin, const char* emb_model_param);
#endif

    int detect(cv::Mat& src, std::vector<FaceBox>& boxes);

private:
    ncnn::Net net_;
    int input_size_ = 192;
    const std::string net_input_name_ = "data";
    const std::string net_output_name_ = "detection_out";
    ncnn::Option option_;
    float threshold_;
    const float mean_val_[3] = {104.f, 117.f, 123.f};
    int thread_num_;
    int min_face_size_;
};



#endif //LIVEBODYEXAMPLE_FACE_DETECTOR_H
