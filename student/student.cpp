#include <fstream>
#include <iostream>
#include <thread>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "cv.h"  
#include "student/student.hpp"
#include "student/behavior.hpp"
#include "student/functions.hpp"
#include<cmath>
#include<tuple>

using namespace cv;
using namespace std;
using namespace caffe;
vector<Class_Info>class_info_all;
vector<int>student_valid;
vector<vector<Student_Info>>students_all(70);
Student_Info biggest_energy;
int standard_frame = 25;
int max_student_num = 0;

void GetStandaredFeats(Net &net1, PoseInfo &pose,Mat &frame,int &n,string &output){
	if (n%standard_frame == 0){
		Timer timer;
		pose_detect(net1, frame, pose);
		if (pose.subset.size() > max_student_num){
			max_student_num = pose.subset.size();
			cout << pose.subset.size() << endl;
			student_valid.clear();
			for (int i = 0; i < 70; i++){
				students_all[i].clear();
			}		
			for (int i = 0; i < pose.subset.size(); i++){
				float score = float(pose.subset[i][18]) / pose.subset[i][19];
				if (pose.subset[i][19] >= 3 && score >= 0.4){
					if (pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						float wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						float wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						float wid = MAX(wid1, wid2);
						if (wid == 0)continue;
						
						Rect standard_rect;
						standard_rect.x = pose.candicate[pose.subset[i][1]][0] - wid - 5;
						standard_rect.y = pose.candicate[pose.subset[i][1]][1];
						standard_rect.width = wid1 + wid2 + 10;
						standard_rect.height = wid1 + wid2-15;
						if (standard_rect.height < 5)standard_rect.height = 15;
						refine(standard_rect, frame);
						//cv::rectangle(frame, standard_rect, Scalar(0, 0, 255), 2, 8, 0);
					
						Student_Info student_ori;
						student_ori.body_bbox = standard_rect;
						student_ori.neck_loc = Point2f(pose.candicate[pose.subset[i][1]][0], pose.candicate[pose.subset[i][1]][1]);
		
						if (pose.subset[i][0] != -1){
							student_ori.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
							student_ori.front = true;
						}
						else{
							student_ori.loc = student_ori.neck_loc;
							student_ori.front = false;
						}
						student_valid.push_back(i);
						students_all[i].push_back(student_ori);					
						string b = output + "/" + "0.jpg";
						//cv::circle(frame, student_ori.loc, 3, cv::Scalar(0, 0, 255), -1);
						cv::putText(frame, to_string(i), student_ori.loc, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 1);
						cv::imwrite(b, frame);
					}
				}
			}
		}
	}
}
std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose, string &output){
	/*vector<Student_Info>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output)*/
	Timer timer;
	
	if (n % standard_frame == 0){

		/*char buf[300];
		sprintf(buf, "../tmp/%d.jpg", n);
		cv::imwrite(buf, image);*/
		/*Mat image1;
		image.copyTo(image1);*/
		timer.Tic();
		pose_detect(net1, image, pose);
		/*timer.Toc();
		cout << "pose detect cost " << timer.Elasped() / 1000.0 << " s" << endl;
		timer.Tic();*/
		int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };
		int x[18];
		int y[18];

		for (int i = 0; i < pose.subset.size(); i++){
			Student_Info student_info;
			student_info.cur_frame1 = n;
			
			int symbol_raise = 0;
			int v = 0;
			float score = float(pose.subset[i][18]) / pose.subset[i][19];
			if (pose.subset[i][19] >= 3 && score >= 0.4){
				for (int j = 0; j < 8; j++){
					if (pose.subset[i][j] == -1){
						x[j] = 0;
						y[j] = 0;
						v = 1;
						if (j == 0){
							v = 0;
						}
					}
					else{
						x[j] = pose.candicate[pose.subset[i][j]][0];
						y[j] = pose.candicate[pose.subset[i][j]][1];
					}			
					//student_info.all_points.push_back(Point2f(x[j], y[j]));
				}

				//-----------判断双手垂直（为了站立）--------------------

				if (v == 0){
					float angle_r = CalculateVectorAngle(x[2], y[2], x[3], y[3], x[4], y[4]);
					float angle_l = CalculateVectorAngle(x[5], y[5], x[6], y[6], x[7], y[7]);
					bool Vertical_l = false;
					bool Vertical_r = false;
					if ((y[4] > y[3] && y[3] > y[2]) && (y[7] > y[6] && y[6] > y[5])){
						float longer_limb = max(abs(y[4] - y[2]), abs(y[7] - y[5]));
						float shorter_limb = min(abs(y[4] - y[2]), abs(y[7] - y[5]));
						/*float longer_width = max(abs(x[5] - x[2]), abs(x[7] - x[4]));
						float shorter_width = min(abs(x[5] - x[2]), abs(x[7] - x[4]));*/
						//if (shorter_limb / longer_limb > 0.75/* && shorter_width / longer_width > 0.7*/){
							if (abs(y[4] - y[3]) >= abs(x[4] - x[3]) && abs(y[2] - y[3]) >= abs(x[2] - x[3])){
								if (float(y[4] - y[3]) / float(y[3] - y[2]) > 0.7 && (angle_r > 135 && angle_l > 115)){
									Vertical_r = true;
								}
							}
							if (abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5])){
								if (float(y[7] - y[6]) / float(y[6] - y[5]) > 0.7 && (angle_l > 135 && angle_r > 115)){
									Vertical_l = true;
								}
							}
						//}
					}
					if ((Vertical_r || Vertical_l)){
						student_info.arm_vertical = true;
						//cv::putText(image, status6, cv::Point2f(x[1], y[1]), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 255), 1);
					}
				}

				//----------------判断举手-----------------------------
				if (pose.subset[i][0] != -1){
					if (pose.subset[i][4] != -1 && pose.subset[i][3] != -1 && pose.subset[i][2] != -1){
						if (y[4] <= y[3] && y[3] <= y[2] && (y[3] - y[4] > 10)){
							symbol_raise = 1;
						}
						else if (y[2]-y[4]>10)symbol_raise = 1;
					}
					if (pose.subset[i][7] != -1 && pose.subset[i][6] != -1 && pose.subset[i][5] != -1){
						if (y[7] <= y[6] && y[6] <= y[5] && (y[6] - y[7] > 10)){
							symbol_raise = 1;
						}
						else if (y[5]-y[7]>10)symbol_raise = 1;
					}
				}
				
				/*if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][4] != -1){
					if (y[0] < y[1] && y[4] < y[0] && (y[0] - y[4])>10)symbol_raise = 1;
					}
					if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][7] != -1){
					if (y[0] < y[1] && y[7] < y[0] && (y[0] - y[7])>10)symbol_raise = 1;
					}*/
				if (symbol_raise == 1){    //如果举手
					student_info.raising_hand = true;
				}
				//-----------------判断扭头 判断转身 判断背身（为了讨论）判断低头------------------------------

				if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
					if (x[2] < x[5]){
						if (x[0] >= x[5]){
							student_info.turn_head = true;
						}
						else{
							if (abs(x[0] - x[1]) / abs(x[0] - x[5]) > 5)student_info.turn_head = true;
						}
						if (x[0] <= x[2]){
							student_info.turn_head = true;
						}
						else{
							if (abs(x[0] - x[1]) / abs(x[0] - x[2]) > 5)student_info.turn_head = true;
						}
					}
					else{
						student_info.turn_head = true;
					}
				}
				if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
					if (abs(y[2] - y[5]) >= abs(x[2] - x[5])){
						student_info.turn_body = true;
					}
					if (x[2] >= x[5])student_info.back = true;
					if (pose.subset[i][0] == -1)student_info.bow_head_tmp = true;
				}

				//-------------------判断低头--------------------------
				if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1){

					if (y[1] > image.size().height / 2){
						if (y[0] - y[1] >= 10)student_info.bow_head_tmp = true;
					}
					else if (y[0] > y[1])student_info.bow_head_tmp = true;

				}


				//--------------------use IOU to classify--------------------------------------------------------------------------------

				//----------------obtain a rect range for i person in a new frame---------------------
				if (/*pose.subset[i][0] != -1&&*/pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){

					float wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
					float wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
					float wid = MAX(wid1, wid2);
					if (wid == 0)continue;

					Rect cur_rect;
					if (student_info.arm_vertical){
						cur_rect.x = pose.candicate[pose.subset[i][1]][0] - wid-5;
						cur_rect.y = pose.candicate[pose.subset[i][1]][1];
						cur_rect.width = wid1 + wid2+10;
						cur_rect.height = wid1 + wid2 -15+ 40;
					}
					else{
						cur_rect.x = pose.candicate[pose.subset[i][1]][0] - wid;
						cur_rect.y = pose.candicate[pose.subset[i][1]][1];
						cur_rect.width = wid1 + wid2;
						cur_rect.height = wid1 + wid2 - 15;
						if (cur_rect.height < 5)cur_rect.height = 15;
					}
					refine(cur_rect, image);

					//cv::rectangle(image, cur_rect, Scalar(0, 255, 0), 2, 8, 0);

					student_info.body_bbox = cur_rect;
					if (cur_rect.y < image.size().height*0.3 && cur_rect.height>80){
						cur_rect = Rect(0, 0, 1, 1);
					}
					student_info.neck_loc = Point2f(pose.candicate[pose.subset[i][1]][0], pose.candicate[pose.subset[i][1]][1]);
					if (pose.subset[i][0] != -1){
						student_info.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
						student_info.front = true;
					}
					else{
						student_info.loc = student_info.neck_loc;
						student_info.front = false;
					}

					//---------------- IOU recognization in a rect range --------------------------------------

					std::multimap<float, int, greater<float>>IOU_map;
					for (int j = 0; j < student_valid.size(); j++){
						float cur_IOU = Compute_IOU(students_all[student_valid[j]][0].body_bbox, cur_rect);
						IOU_map.insert(make_pair(cur_IOU, student_valid[j]));
					}
					if (IOU_map.begin()->first > 0){
						int thre = (student_info.loc.y < image.size().height / 2) ? 100 : 150;
						int size1 = students_all[IOU_map.begin()->second].size();
						float dis = abs(students_all[IOU_map.begin()->second][size1 - 1].loc.y - student_info.loc.y);
						if (dis < thre){
							students_all[IOU_map.begin()->second].push_back(student_info);
						}
					}
					else{
						//cv::rectangle(image,student_info.body_bbox,Scalar(0,255,0),1,8,0);
						students_all[69].push_back(student_info);
					}
				}
				/*for (int j = 0; j < 8; j++){
					if (!(x[j] || y[j])){
						continue;
					}
					else{
						cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
					}
				}*/
			} //if (pose.subset[i][19] >= 3 && score >= 0.4) end
		}//for (int i = 0; i < pose.subset.size(); i++) end

		//----------------------分析行为------------------------------
		Analys_Behavior(students_all, student_valid, class_info_all, image, n);
		//writeJson(student_valid, students_all, class_info_all, videoname,n);
		//drawGrid(image,student_valid,students_all);

		string output1;
		if (class_info_all.size()>0 && (class_info_all[class_info_all.size() - 1].all_bow_head || class_info_all[class_info_all.size() - 1].all_disscussion_2 || class_info_all[class_info_all.size() - 1].all_disscussion_4)){
			output1 = output + "/" + to_string(n)+"-class" + ".jpg";
		}
		else{
			output1 = output + "/" + to_string(n) + ".jpg";
		}
		cv::imwrite(output1, image);
		timer.Toc();
		cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;
	} //if (n % standard_frame == 0) end
	return std::make_tuple(students_all, class_info_all);
}
