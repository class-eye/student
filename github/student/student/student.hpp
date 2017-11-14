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
	bool turn_head = false;
	Point2f loc;
	Point2f neck_loc;
	Rect body_bbox;
	string output_body_dir;
	bool front;
};
extern vector<string>output_body;
vector<Student_Info> student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose);
void GetStandaredFeats(Net &net1, PoseInfo &pose,Mat &frame,int &n);
#endif