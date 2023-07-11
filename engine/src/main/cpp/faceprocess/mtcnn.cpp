#include <opencv2/opencv.hpp>
#include <chrono>
#include "ncnn/cpu.h"

#include "mtcnn.h"
#include "../android_log.h"
//
// Created by Lonqi on 2017/11/18.
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------

bool cmpScore(Bbox lsh, Bbox rsh)
{
  return (lsh.score < rsh.score);
}
//----------------------------------------------------------------------------------------
bool cmpArea(Bbox lsh, Bbox rsh)
{
  return !(lsh.area < rsh.area);
}

static bool AreaComp(FaceBox &l, FaceBox &r)
{
  return ((l.x2 - l.x1 + 1) * (l.y2 - l.y1 + 1)) > ((r.x2 - r.x1 + 1) * (r.y2 - r.y1 + 1));
}

//----------------------------------------------------------------------------------------
MTCNN::MTCNN() {}
//----------------------------------------------------------------------------------------
MTCNN::~MTCNN()
{
  Pnet.clear();
  Rnet.clear();
  Onet.clear();
}

#if __ANDROID_API__ >= 9
int MTCNN::loadModel(AAssetManager *assetManager)
{
  ncnn::set_cpu_powersave(2);
  ncnn::Option option_;
  option_.lightmode = true;
  option_.num_threads = 2;

  int ret = 0;
  Pnet.opt = option_;
  ret += Pnet.load_param(assetManager, "detection/mtcnn/det1.param");
  ret += Pnet.load_model(assetManager, "detection/mtcnn/det1.bin");
  Rnet.opt = option_;
  ret += Rnet.load_param(assetManager, "detection/mtcnn/det2.param");
  ret += Rnet.load_model(assetManager, "detection/mtcnn/det2.bin");
  Onet.opt = option_;
  ret += Onet.load_param(assetManager, "detection/mtcnn/det3.param");
  ret += Onet.load_model(assetManager, "detection/mtcnn/det3.bin");

  if(ret != 0) {
    LOG_ERR("MTCNN load model failed. %d", ret);
    return -1;
  }

  return 0;


//  int ret = net_.load_param(assetManager, "detection/detection.param");
//  if(ret != 0) {
//      LOG_ERR("FaceDetector load param failed. %d", ret);
//      return -1;
//  }
//
//  ret = net_.load_model(assetManager, "detection/detection.bin");
//  if(ret != 0) {
//      LOG_ERR("FaceDetector load model failed. %d", ret);
//      return -2;
//  }
//  return 0;
}
#else
int MTCNN::loadModel(
  const char *model_bin1, const char *model_param1,
  const char *model_bin2, const char *model_param2,
  const char *model_bin3, const char *model_param3)
{
  ncnn::set_cpu_powersave(2);
  ncnn::Option option_;
  option_.lightmode = true;
  option_.num_threads = 2;

  Pnet.opt = option_;
  Pnet.load_param(model_param1);
  Pnet.load_model(model_bin1);
  Rnet.opt = option_;
  Rnet.load_param(model_param2);
  Rnet.load_model(model_bin2);
  Onet.opt = option_;
  Onet.load_param(model_param3);
  Onet.load_model(model_bin3);

  return 0;
}
#endif
//----------------------------------------------------------------------------------------
void MTCNN::setMinFace(int minSize)
{
  minsize = minSize;
}
//----------------------------------------------------------------------------------------
void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox> &boundingBox_, float scale)
{
  const int stride = 2;
  const int cellsize = 12;
  // score p
  float *p = score.channel(1); // score.data + score.cstep;
  // float *plocal = location.data;
  Bbox bbox;
  float inv_scale = 1.0f / scale;
  for (int row = 0; row < score.h; row++)
  {
    for (int col = 0; col < score.w; col++)
    {
      if (*p > threshold[0])
      {
        bbox.score = *p;
        bbox.x1 = round((stride * col + 1) * inv_scale);
        bbox.y1 = round((stride * row + 1) * inv_scale);
        bbox.x2 = round((stride * col + 1 + cellsize) * inv_scale);
        bbox.y2 = round((stride * row + 1 + cellsize) * inv_scale);
        bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
        const int index = row * score.w + col;
        for (int channel = 0; channel < 4; channel++)
        {
          bbox.regreCoord[channel] = location.channel(channel)[index];
        }
        boundingBox_.push_back(bbox);
      }
      p++;
      // plocal++;
    }
  }
}
//----------------------------------------------------------------------------------------
void MTCNN::nmsTwoBoxs(vector<Bbox> &boundingBox_, vector<Bbox> &previousBox_, const float overlap_threshold, string modelname)
{
  if (boundingBox_.empty())
  {
    return;
  }
  sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
  float IOU = 0;
  float maxX = 0;
  float maxY = 0;
  float minX = 0;
  float minY = 0;
  // std::cout << boundingBox_.size() << " ";
  for (std::vector<Bbox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++)
  {
    for (std::vector<Bbox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();)
    {
      int i = itx - boundingBox_.begin();
      int j = ity - previousBox_.begin();
      maxX = std::max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
      maxY = std::max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
      minX = std::min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
      minY = std::min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
      // maxX1 and maxY1 reuse
      maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
      maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
      // IOU reuse for the area of two bbox
      IOU = maxX * maxY;
      if (!modelname.compare("Union"))
        IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
      else if (!modelname.compare("Min"))
      {
        IOU = IOU / ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area : previousBox_.at(j).area);
      }
      if (IOU > overlap_threshold && boundingBox_.at(i).score > previousBox_.at(j).score)
      {
        // if (IOU > overlap_threshold) {
        itx = boundingBox_.erase(itx);
      }
      else
      {
        itx++;
      }
    }
  }
}
//----------------------------------------------------------------------------------------
void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname)
{
  if (boundingBox_.empty())
  {
    return;
  }
  sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
  float IOU = 0;
  float maxX = 0;
  float maxY = 0;
  float minX = 0;
  float minY = 0;
  std::vector<int> vPick;
  int nPick = 0;
  std::multimap<float, int> vScores;
  const int num_boxes = boundingBox_.size();
  vPick.resize(num_boxes);
  for (int i = 0; i < num_boxes; ++i)
  {
    vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
  }
  while (vScores.size() > 0)
  {
    int last = vScores.rbegin()->second;
    vPick[nPick] = last;
    nPick += 1;
    for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();)
    {
      int it_idx = it->second;
      maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
      maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
      minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
      minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
      // maxX1 and maxY1 reuse
      maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
      maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
      // IOU reuse for the area of two bbox
      IOU = maxX * maxY;
      if (!modelname.compare("Union"))
        IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
      else if (!modelname.compare("Min"))
      {
        IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
      }
      if (IOU > overlap_threshold)
      {
        it = vScores.erase(it);
      }
      else
      {
        it++;
      }
    }
  }

  vPick.resize(nPick);
  std::vector<Bbox> tmp_;
  tmp_.resize(nPick);
  for (int i = 0; i < nPick; i++)
  {
    tmp_[i] = boundingBox_[vPick[i]];
  }
  boundingBox_ = tmp_;
}
//----------------------------------------------------------------------------------------
void MTCNN::refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square)
{
  if (vecBbox.empty())
  {
    cout << "Bbox is empty!!" << endl;
    return;
  }
  float bbw = 0, bbh = 0, maxSide = 0;
  float h = 0, w = 0;
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  for (vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
  {
    bbw = (*it).x2 - (*it).x1 + 1;
    bbh = (*it).y2 - (*it).y1 + 1;
    x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
    y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
    x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
    y2 = (*it).y2 + (*it).regreCoord[3] * bbh;
    if (square)
    {
      w = x2 - x1 + 1;
      h = y2 - y1 + 1;
      maxSide = (h > w) ? h : w;
      x1 = x1 + w * 0.5 - maxSide * 0.5;
      y1 = y1 + h * 0.5 - maxSide * 0.5;
      (*it).x2 = round(x1 + maxSide - 1);
      (*it).y2 = round(y1 + maxSide - 1);
      (*it).x1 = round(x1);
      (*it).y1 = round(y1);
    }
    // boundary check
    if ((*it).x1 < 0)
      (*it).x1 = 0;
    if ((*it).y1 < 0)
      (*it).y1 = 0;
    if ((*it).x2 > width)
      (*it).x2 = width - 1;
    if ((*it).y2 > height)
      (*it).y2 = height - 1;

    it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
  }
}
//----------------------------------------------------------------------------------------
void MTCNN::PNet(float scale)
{
  // first stage
  int hs = (int)ceil(img_h * scale);
  int ws = (int)ceil(img_w * scale);
  ncnn::Mat in;
  resize_bilinear(img, in, ws, hs);
  ncnn::Extractor ex = Pnet.create_extractor();
  ex.set_light_mode(true);
  // sex.set_num_threads(4);
  ex.input("data", in);
  ncnn::Mat score_, location_;
  ex.extract("prob1", score_);
  ex.extract("conv4-2", location_);
  std::vector<Bbox> boundingBox_;

  generateBbox(score_, location_, boundingBox_, scale);
  nms(boundingBox_, nms_threshold[0]);

  firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
  boundingBox_.clear();
}
//----------------------------------------------------------------------------------------
void MTCNN::PNet()
{
  firstBbox_.clear();
  float minl = img_w < img_h ? img_w : img_h;
  float m = (float)MIN_DET_SIZE / minsize;
  minl *= m;
  float factor = pre_facetor;
  vector<float> scales_;
  while (minl > MIN_DET_SIZE)
  {
    scales_.push_back(m);
    minl *= factor;
    m = m * factor;
  }
  for (size_t i = 0; i < scales_.size(); i++)
  {
    int hs = (int)ceil(img_h * scales_[i]);
    int ws = (int)ceil(img_w * scales_[i]);
    ncnn::Mat in;
    resize_bilinear(img, in, ws, hs);
    ncnn::Extractor ex = Pnet.create_extractor();
    // ex.set_num_threads(2);
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat score_, location_;
    ex.extract("prob1", score_);
    ex.extract("conv4-2", location_);
    std::vector<Bbox> boundingBox_;
    generateBbox(score_, location_, boundingBox_, scales_[i]);
    nms(boundingBox_, nms_threshold[0]);
    firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
    boundingBox_.clear();
  }
}
//----------------------------------------------------------------------------------------
void MTCNN::RNet()
{
  secondBbox_.clear();
  for (vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++)
  {
    ncnn::Mat tempIm;
    copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
    ncnn::Mat in;
    resize_bilinear(tempIm, in, 24, 24);
    ncnn::Extractor ex = Rnet.create_extractor();
    // ex.set_num_threads(2);
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat score, bbox;
    ex.extract("prob1", score);
    ex.extract("conv5-2", bbox);
    if ((float)score[1] > threshold[1])
    {
      for (int channel = 0; channel < 4; channel++)
      {
        it->regreCoord[channel] = (float)bbox[channel]; //*(bbox.data+channel*bbox.cstep);
      }
      it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
      //			it->score = score.channel(1)[0];//*(score.data+score.cstep);
      it->score = score[1]; //*(score.data+score.cstep);
      secondBbox_.push_back(*it);
    }
  }
}
//----------------------------------------------------------------------------------------
void MTCNN::ONet(void)
{
  thirdBbox_.clear();
  for (vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++)
  {
    ncnn::Mat tempIm;
    copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
    ncnn::Mat in;
    resize_bilinear(tempIm, in, 48, 48);
    ncnn::Extractor ex = Onet.create_extractor();
    // ex.set_num_threads(2);
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat score, bbox, keyPoint;
    ex.extract("prob1", score);
    ex.extract("conv6-2", bbox);
    ex.extract("conv6-3", keyPoint);
    if ((float)score[1] > threshold[2])
    {
      for (int channel = 0; channel < 4; channel++)
      {
        it->regreCoord[channel] = (float)bbox[channel];
      }
      it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
      //			it->score = score.channel(1)[0];
      it->score = score[1]; //*(score.data+score.cstep);
      for (int num = 0; num < 5; num++)
      {
        //(it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
        it->landmark.x[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
        //(it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
        it->landmark.y[num] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
      }
      thirdBbox_.push_back(*it);
    }
  }
}
//----------------------------------------------------------------------------------------
int MTCNN::detect(cv::Mat &bgr, std::vector<FaceBox> &boxes)
{
  auto c_detect = std::chrono::steady_clock::now();
  int width = bgr.cols;
  int height = bgr.rows;
  const int target_size = 192;
  int w = width;
  int h = height;
  float scale = 1.f;
  if (w > h)
  {
    scale = (float)target_size / w;
    w = target_size;
    h = h * scale;
  }
  else
  {
    scale = (float)target_size / h;
    h = target_size;
    w = w * scale;
  }

  ncnn::Mat img_ = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
  img = img_;
  img_w = img.w;
  img_h = img.h;
  img.substract_mean_normalize(mean_vals, norm_vals);

  // the first stage's nms
  PNet();
  if (firstBbox_.size() < 1)
    return -1;
  nms(firstBbox_, nms_threshold[0]);
  refine(firstBbox_, img_h, img_w, true);

  // second stage
  RNet();
  if (secondBbox_.size() < 1)
    return -1;
  nms(secondBbox_, nms_threshold[1]);
  refine(secondBbox_, img_h, img_w, true);

  // third stage
  ONet();
  if (thirdBbox_.size() < 1)
    return -1;
  refine(thirdBbox_, img_h, img_w, true);
  nms(thirdBbox_, nms_threshold[2], "Min");

  boxes.resize(thirdBbox_.size());
  for (size_t i = 0; i < thirdBbox_.size(); i++)
  {
    FaceObject obj;
    obj.rect.x = thirdBbox_[i].x1 / scale;
    obj.rect.y = thirdBbox_[i].y1 / scale;
    obj.rect.width = (thirdBbox_[i].x2 - thirdBbox_[i].x1) / scale;
    obj.rect.height = (thirdBbox_[i].y2 - thirdBbox_[i].y1) / scale;
    cv::rectangle(bgr, obj.rect, cv::Scalar(0, 255, 0), 3);

    for (int n = 0; n < 5; n++)
    {
      obj.landmark[n].x = thirdBbox_[i].landmark.x[n] / scale;
      obj.landmark[n].y = thirdBbox_[i].landmark.y[n] / scale;
    }
    for (int i = 0; i < 5; i++)
      cv::circle(bgr, obj.landmark[i], 2, cv::Scalar(255, 0, 0), 2);

    obj.prob = thirdBbox_[i].score;

    FaceBox box;
    box.confidence = obj.prob;
    box.x1 = obj.rect.x;
    box.y1 = obj.rect.y;
    box.x2 = box.x1 + obj.rect.width;
    box.y2 = box.y1 + obj.rect.height;
    boxes.emplace_back(box);
  }

  // sort
  std::sort(boxes.begin(), boxes.end(), AreaComp);

  auto c_end = std::chrono::steady_clock::now();
  float t_detect = std::chrono::duration_cast <std::chrono::milliseconds> (c_end - c_detect).count();
  LOG_INFO("detect time: %.3f ms\n", t_detect);

  return 0;
}
//----------------------------------------------------------------------------------------
