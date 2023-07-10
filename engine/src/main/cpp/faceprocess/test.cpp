#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

#include "definition.h"
#include "face_detector.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  Mat frame;
  FaceDetector* detector = new FaceDetector();
  detector->LoadModel("/research/android/Silent-Face-Anti-Spoofing-APK/engine/src/main/assets/detection/detection.bin", "//research/android/Silent-Face-Anti-Spoofing-APK/engine/src/main/assets/detection/detection.param");
  VideoCapture cap("rtsp://127.0.0.1:8554/stream1");

  if (!cap.isOpened()) {
    cerr << "ERROR: Unable to open the camera" << endl;
    return 0;
  }
  cout << "Start grabbing, press ESC on Live window to terminate" << endl;
  while(1) {
    cap >> frame;
    if (frame.empty()) {
      cerr << "ERROR: Unable to grab from the camera" << endl;
      break;
    }
    std::vector<FaceBox> bboxes;
    detector->Detect(frame, bboxes);
    for (size_t i = 0; i < bboxes.size(); i++ ) {
      FaceBox& r = bboxes[i];
      // std::vector<cv::Point2f> lmks;
      // process_ctx->getLmk(frame, r, lmks);
      // for (auto&pt : lmks) {
      //   cv::circle(frame, cv::Point2f(pt.x + r.x, pt.y + r.y), 1, cv::Scalar(255, 0, 0), 1);
      // }
      rectangle(frame, cvPoint(cvRound(r.x1), cvRound(r.y1)),
                    cvPoint(cvRound(r.x2), cvRound(r.y2)),
                    Scalar(0, 255, 0), 3);
    }
    //show output
    imshow("Test face processing flow!!!", frame);
    char esc = waitKey(5);
    if(esc == 27) break;
  }
  destroyAllWindows();
  return 0;
}