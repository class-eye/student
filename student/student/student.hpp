#ifndef _STUDENT_H_
#define _STUDENT_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "pose.hpp"
#include "hand.hpp"
#include "Timer.hpp"
#include "fs.hpp"
#include "incCn/HCNetSDK.h"  
#include "incCn/PlayM4.h" 
using namespace cv;
using namespace caffe;
struct Class_Info{
	bool all_bow_head=false;
	bool all_disscussion_2 = false;
	bool all_disscussion_4 = false;
	int cur_frame=0;
	PLAYM4_SYSTEM_TIME pstSystemTime;
};
struct Student_Info{
	bool raising_hand=false;
	bool standing=false;
	bool disscussion= false;
	bool daze = false;
	bool bow_head = false;
	bool bow_head_each = false;

	bool turn_head = false;
	bool arm_vertical = false;
	bool whisper = false;
	bool turn_body = false;
	bool bow_head_tmp = false;
	Point2f loc;
	Point2f neck_loc;
	Rect body_bbox;
	Rect body_for_save;
	//string output_body_dir;
	int away_from_seat = 0;
	int cur_frame1=0;
	int cur_size = 0;
	int energy = 0;
	int max_energy = 0;
	bool front=false;
	bool back = false;
	vector<int>miss_frame;
	//vector<Point2f>all_points;

	bool real_raise = false;
	float scores = 0.0;

	PLAYM4_SYSTEM_TIME pstSystemTime;
};
//vector<Student_Info> student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output);
std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(Net &net1, Net &net2, Mat &image, int &n, PoseInfo &pose, string &output, PLAYM4_SYSTEM_TIME &pstSystemTime, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info_all);
void GetStandaredFeats(Net &net1, PoseInfo &pose, Mat &frame, int &n, string &output, int &max_student_num, vector<vector<Student_Info>>&students_all, vector<int>&student_valid,vector<Class_Info>&class_info_all);

#endif