#ifndef SCRFD_H
#define SCRFD_H

#include <opencv2/core/core.hpp>
#include "ncnn/net.h"

#include "definition.h"

class SCRFD {
public:

#if __ANDROID_API__ >= 9
  int loadModel(AAssetManager* assetManager);
#else
  int loadModel(const char* model_bin, const char* model_param);
#endif

  int detect(cv::Mat& src, std::vector<FaceBox>& boxes, float prob_threshold=0.5f, float nms_threshold=0.45f);

private:
  ncnn::Net scrfd;
};

#endif //SCRFD_H