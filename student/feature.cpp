#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include "feature/feature.hpp"
using namespace caffe;
using cv::Mat;
using cv::Rect;
using cv::Point2f;
using std::vector;
using std::string;




void Extract(Net &net,const Mat& img, FaceInfo& face) {
  // we need gray image
  CV_Assert(img.type() == CV_8UC1);
  //Mat data = GetRefinedFace(img, face, cv::Size(128, 128), 48, 40);
  Mat data;
  cv::resize(img, data, cv::Size(128, 128));
  data.convertTo(data, CV_32F, 1.f / 255.f);
  shared_ptr<Blob> input = net.blob_by_name("data");
  const int kBytes = input->offset(1) * sizeof(float);
  memcpy(input->mutable_cpu_data(), data.data, kBytes);
  net.Forward();
  shared_ptr<Blob> feature = net.blob_by_name("eltwise_fc1");
  const int kFeatureSize = feature->channels();
  face.feature.resize(kFeatureSize);
  for (int i = 0; i < kFeatureSize; i++) {
    face.feature[i] = feature->data_at(0, i, 0, 0);
  }
}

//void Extract(Net &net,const Mat& img, vector<FaceInfo>& faces) {
//  // we need gray image
//  CV_Assert(img.type() == CV_8UC1);
//  for (int i = 0; i < faces.size(); i++) {
//    Extract(net,img, faces[i]);
//  }
//}

