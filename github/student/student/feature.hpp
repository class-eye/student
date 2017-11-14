#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>

#include <string>
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;
struct FaceInfo {
	std::vector<float> feature;
	Point2f person_loc;
	Rect face_bbox;
	string faced;
	Point2f nose_loc;
	int raising_num=0;
	int standing_num=0;
	int student_id;
};
void Extract(Net &net, const Mat& img, FaceInfo& face);
//void Extract(Net &net, const Mat& img, vector<FaceInfo>& faces);
#endif