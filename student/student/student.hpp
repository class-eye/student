#ifndef _STUDENT_H_
#define _STUDENT_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "pose.hpp"
#include "Timer.hpp"
#include "fs.hpp"

using namespace cv;
using namespace caffe;
struct Class_Info{
	bool all_bow_head=false;
	bool all_disscussion_2 = false;
	bool all_disscussion_4 = false;
	int cur_frame;
};
struct Student_Info{
	bool raising_hand=false;
	bool standing=false;
	bool disscussion= false;
	bool daze = false;
	bool bow_head = false;

	bool turn_head = false;
	bool arm_vertical = false;
	bool whisper = false;
	bool turn_body = false;
	bool bow_head_tmp = false;
	Point2f loc;
	Point2f neck_loc;
	Rect body_bbox;
	string output_body_dir;
	int cur_frame1;
	int cur_size = 0;
	int energy = 0;
	int max_energy = 0;
	bool front=false;
	bool back = false;
	vector<Point2f>all_points;
};
//vector<Student_Info> student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output);
std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose, string &output);
void GetStandaredFeats(Net &net1, PoseInfo &pose,Mat &frame,int &n,string &output);
extern string videoname;
#endif