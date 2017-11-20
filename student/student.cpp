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
using namespace cv;
using namespace std;
using namespace caffe;

vector<int>student_valid;
vector<vector<Student_Info>>students_all(70);
int standard_frame = 25;
int max_student_num = 0;
Class_Info class_info;
void GetStandaredFeats(Net &net1, PoseInfo &pose,Mat &frame,int &n){
	if (n%standard_frame == 0){
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
					if (/*pose.subset[i][0] != -1&&*/pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						float wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						float wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						float wid = MAX(wid1, wid2);
						if (wid == 0)continue;
						
						Rect standard_rect;
						standard_rect.x = pose.candicate[pose.subset[i][1]][0]-wid;
						standard_rect.y = pose.candicate[pose.subset[i][1]][1];
						standard_rect.width = wid1 + wid2;
						standard_rect.height = wid1 + wid2-15;
						refine(standard_rect, frame);
						cv::rectangle(frame, standard_rect, Scalar(0, 0, 255), 2, 8, 0);
						/*if (!fs::IsExists(output_body[i])){
							fs::MakeDir(output_body[i]);
							}*/
						
						Student_Info student_ori;
						student_ori.body_bbox = standard_rect;
						student_ori.neck_loc = Point2f(pose.candicate[pose.subset[i][1]][0], pose.candicate[pose.subset[i][1]][1]);
						/*if (pose.subset[i][0] != -1){
							student_ori.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
							}*/
						if (pose.subset[i][0] != -1){
							student_ori.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
							student_ori.front = true;
						}
						else{
							student_ori.loc = student_ori.neck_loc;
							student_ori.front = false;
						}
						
						/*student_ori.output_body_dir = output_body[i];*/

						student_valid.push_back(i);
						students_all[i].push_back(student_ori);

						//writeJson(n, face_ori.feature, out);
						//string b = student_ori.output_body_dir + "/" + "0.jpg";
						string outputdir = "../output";
						string b = outputdir + "/" + "0.jpg";
						cv::circle(frame, student_ori.loc, 3, cv::Scalar(0, 0, 255), -1);
						cv::imwrite(b, frame);
					}
				}
			}
		}
	}
}

vector<Student_Info>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose){
	Timer timer;
	vector<Student_Info>student_have_action;
	if (n % standard_frame == 0){
		
			/*char buf[300];
			sprintf(buf, "../tmp/%d.jpg", n);
			cv::imwrite(buf, image);*/
			/*Mat image1;
			image.copyTo(image1);*/
			timer.Tic();
			pose_detect(net1, image, pose);
			/*timer.Toc();
			cout << "pose detect cost " << timer.Elasped() / 1000.0 << " s" << endl;*/
			//timer.Tic();
			int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };
			int x[18];
			int y[18];				
		
			for (int i = 0; i < pose.subset.size(); i++){
				Student_Info student_info;
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
					}
					//-----------�ж�˫�ִ�ֱ��Ϊ��վ����--------------------

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
							if (shorter_limb / longer_limb > 0.75/* && shorter_width / longer_width > 0.7*/){
								if (abs(y[4] - y[3]) >= abs(x[4] - x[3]) && abs(y[2] - y[3]) >= abs(x[2] - x[3])){
									if (float(y[4] - y[3]) / float(y[3] - y[2]) > 0.7 && (angle_r > 145 && angle_l > 115)){
										Vertical_r = true;
									}
								}
								if (abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5])){
									if (float(y[7] - y[6]) / float(y[6] - y[5]) > 0.7 && (angle_l > 145 && angle_r > 115)){
										Vertical_l = true;
									}
								}
							}
						}
						if ((Vertical_r || Vertical_l)){
							student_info.arm_vertical = true;
							//cv::putText(image, status6, cv::Point2f(x[1], y[1]), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 255), 1);
						}
					}

					//----------------�жϾ���-----------------------------

					if (pose.subset[i][0] != -1){
						if (pose.subset[i][4] != -1 && pose.subset[i][3] != -1 && pose.subset[i][2] != -1){
							if (y[4] <= y[3] && y[3] <= y[2] && (y[3] - y[4] > 10)){
								symbol_raise = 1;
							}
						}
						if (pose.subset[i][7] != -1 && pose.subset[i][6] != -1 && pose.subset[i][5] != -1){
							if (y[7] <= y[6] && y[6] <= y[5] && (y[6] - y[7] > 10)){
								symbol_raise = 1;
							}
						}
					}
					if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][4] != -1){
						if (y[0] < y[1] && y[4] < y[0] && (y[0] - y[4])>10)symbol_raise = 1;
					}
					if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][7] != -1){
						if (y[0] < y[1] && y[7] < y[0] && (y[0] - y[7])>10)symbol_raise = 1;
					}
					if (symbol_raise == 1){    //�������
						student_info.raising_hand = true;				
					}
					//-----------------�ж�Ťͷ �ж�ת�� �жϱ�����Ϊ�����ۣ��жϵ�ͷ------------------------------

					if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1&& pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						if (x[2] < x[5]){
							if (x[0] >= x[5]){
								student_info.turn_head = true;
							}
							else{
								if (abs(x[0] - x[1]) / abs(x[0] - x[5])>5)student_info.turn_head = true;
							}
							if (x[0] <= x[2]){
								student_info.turn_head = true;
							}
							else{
								if (abs(x[0] - x[1]) / abs(x[0] - x[2])>5)student_info.turn_head = true;
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
						if (x[2] > x[5])student_info.back = true;
						if (pose.subset[i][0] == -1)student_info.bow_head_tmp = true;
					}
				
					//-------------------�жϵ�ͷ--------------------------
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
							cur_rect.x = pose.candicate[pose.subset[i][1]][0]-wid;
							cur_rect.y = pose.candicate[pose.subset[i][1]][1];
							cur_rect.width = wid1 + wid2;
							cur_rect.height = wid1 + wid2 + 45;
						}
						else{
							cur_rect.x = pose.candicate[pose.subset[i][1]][0]-wid;
							cur_rect.y = pose.candicate[pose.subset[i][1]][1];
							cur_rect.width = wid1 + wid2;
							cur_rect.height = wid1 + wid2-15;
							if (cur_rect.height < 0)cur_rect.height = 10;
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
						
						vector<float>dist;
						vector<vector<float>>dists;
						for (int j = 0; j < student_valid.size(); j++){
							float cur_IOU = Compute_IOU(students_all[student_valid[j]][0].body_bbox, cur_rect);
							dist.push_back(student_valid[j]);
							dist.push_back(cur_IOU);
							dists.push_back(dist);
							dist.clear();
						}
				
						sort(dists.begin(), dists.end(), greate2);
						if (dists[0][1] > 0){
							int size1 = students_all[dists[0][0]].size();
							float dis = abs(students_all[dists[0][0]][size1 - 1].loc.y-student_info.loc.y);
							if (dis < 150){
								students_all[dists[0][0]].push_back(student_info);
							}
						}
						//string a = students_all[dists[0][0]][0].output_body_dir + "/" + to_string(n) + "_" + to_string(dists[0][1]) + ".jpg";
						/*Mat bodyimg = image(cur_rect);
						cv::imwrite(a, bodyimg);*/
					}			
					/*for (int j = 0; j < 6; j++){
						if (!(x[j] || y[j])){
							continue;
						}
						else{
							cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
						}
					}*/
				} //if (pose.subset[i][19] >= 3 && score >= 0.4) end	
			}//for (int i = 0; i < pose.subset.size(); i++) end

			Analys_Behavior(students_all,student_valid,class_info,image);

			string outputp1 = "../output/" + to_string(n) + ".jpg";
			cv::imwrite(outputp1, image);
			timer.Toc();
			cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;
		} //if (n % standard_frame == 0) end
	return student_have_action;
}