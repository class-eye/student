#ifndef _STUDENT_H_
#define _STUDENT_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "pose.hpp"
#include "Timer.hpp"
#include "fs.hpp"
#include "feature.hpp"
using namespace cv;
using namespace caffe;
struct Student_Info{
	bool raising_hand=false;
	bool standing=false;
	Point loc;
};
vector<Student_Info> student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose);
#endif