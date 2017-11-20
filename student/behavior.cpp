#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "student/student.hpp"
#include "student/behavior.hpp"
#include<cmath>
using namespace cv;
using namespace std;

void Analys_Behavior(vector<vector<Student_Info>>students_all, vector<int>&student_valid, Class_Info &class_info, Mat &image){

	string status1 = "raising hand";
	string status2 = "standing";
	string status3 = "disscussion";
	string status4 = "daze";
	string status5 = "bow head";
	string status6 = "arm_ver";

	int num_of_back = 0;
	int num_of_disscuss = 0;
	int num_of_bowhead = 0;
	for (int j = 0; j < student_valid.size(); j++){
		int x1 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].neck_loc.x;
		int y1 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].neck_loc.y;

		if (students_all[student_valid[j]].size() <= 5){
			for (int k = 1; k < students_all[student_valid[j]].size() - 1; k++){
				if (students_all[student_valid[j]][k].front == true && students_all[student_valid[j]][k + 1].front == true){
					line(image, Point2f(students_all[student_valid[j]][k].loc.x, students_all[student_valid[j]][k].loc.y), Point2f(students_all[student_valid[j]][k + 1].loc.x, students_all[student_valid[j]][k + 1].loc.y), cv::Scalar(0, 0, 255), 2, 8, 0);
				}
				else{
					line(image, Point2f(students_all[student_valid[j]][k].neck_loc.x, students_all[student_valid[j]][k].neck_loc.y), Point2f(students_all[student_valid[j]][k + 1].neck_loc.x, students_all[student_valid[j]][k + 1].neck_loc.y), cv::Scalar(255, 0, 0), 2, 8, 0);
				}
			}
		}
		else{
			int k1 = students_all[student_valid[j]].size() - 5;
			int k2 = students_all[student_valid[j]].size() - 4;
			int k3 = students_all[student_valid[j]].size() - 3;
			int k4 = students_all[student_valid[j]].size() - 2;
			int k5 = students_all[student_valid[j]].size() - 1;

			if (students_all[student_valid[j]][k1].front == true && students_all[student_valid[j]][k2].front == true && students_all[student_valid[j]][k3].front == true && students_all[student_valid[j]][k4].front == true && students_all[student_valid[j]][k5].front == true)
			{
				for (int k = students_all[student_valid[j]].size() - 5; k < students_all[student_valid[j]].size() - 1; k++){
					line(image, Point2f(students_all[student_valid[j]][k].loc.x, students_all[student_valid[j]][k].loc.y), Point2f(students_all[student_valid[j]][k + 1].loc.x, students_all[student_valid[j]][k + 1].loc.y), cv::Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			else{
				for (int k = students_all[student_valid[j]].size() - 5; k < students_all[student_valid[j]].size() - 1; k++){
					line(image, Point2f(students_all[student_valid[j]][k].neck_loc.x, students_all[student_valid[j]][k].neck_loc.y), Point2f(students_all[student_valid[j]][k + 1].neck_loc.x, students_all[student_valid[j]][k + 1].neck_loc.y), cv::Scalar(255, 0, 0), 2, 8, 0);
				}
			}
			//-------------------�ռ�3s�ڵ���Ϣ---------------------------

			vector<float>nose_range;
			vector<float>box_range;
			vector<float>nose_y;
			for (int k = students_all[student_valid[j]].size() - 3; k < students_all[student_valid[j]].size(); k++){
				nose_range.push_back(students_all[student_valid[j]][k].loc.x);
				nose_y.push_back(students_all[student_valid[j]][k].loc.y);
				box_range.push_back(students_all[student_valid[j]][k].body_bbox.width);
			}
			float max_nose = *max_element(nose_range.begin(), nose_range.end());
			float min_nose = *min_element(nose_range.begin(), nose_range.end());
			float max_nose_y = *max_element(nose_y.begin(), nose_y.end());

			float max_width = (*max_element(box_range.begin(), box_range.end())) * 2 / 3;
			//------------------�ж�����(3s��)-----------------------------


			int count1 = 0;
			if (students_all[student_valid[j]].size() > 10){
				for (int k = students_all[student_valid[j]].size() - 10; k < students_all[student_valid[j]].size(); k++){
					if (students_all[student_valid[j]][k].turn_head == true)count1++;
				}
			}
			if (count1 >= 8 || students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].turn_body == true || students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true){
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion = true;
			}
			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true){
				num_of_back++;
			}
			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion == true){
				num_of_disscuss++;
			}
			if (num_of_back >= 10 || num_of_disscuss >= 10){
				class_info.all_disscussion = true;
			}
			//------------------�жϵ�ͷ-----------------------------------
			int count2 = 0;
			if (students_all[student_valid[j]].size() > 10){
				for (int k = students_all[student_valid[j]].size() - 10; k < students_all[student_valid[j]].size(); k++){
					if (students_all[student_valid[j]][k].bow_head_tmp == true)count2++;
				}
			}
			if (count2 >= 9)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head = true;

			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head == true){
				num_of_bowhead++;
			}
			if (num_of_bowhead >= 10){
				class_info.all_bow_head = true;
			}
			//-----------------�ж�����(3s��)---------------------------
			int thre1;
			if (max_nose_y - nose_y[nose_y.size() - 1] > max_width){
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].arm_vertical == true)
				{
					students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing = true;
				}
			}
			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 2].standing == true){
				float dis = abs(students_all[student_valid[j]][students_all[student_valid[j]].size() - 2].loc.y - students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y);
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y < image.size().height / 5)thre1 = 1;
				else thre1 = 20;
				if (dis < thre1)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing = true;
			}

			int thre;
			int cur_y = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y;
			if (cur_y >= image.size().height / 2)thre = 15;
			else if (cur_y < image.size().height / 2 && cur_y >= image.size().height / 4)thre = 10;
			else thre = 5;

			//-------------------�жϷ���(Ns��)--------------------------
			if (students_all[student_valid[j]].size() > 30){
				vector<float>nose_range_x;
				vector<float>nose_range_y;
				for (int k = students_all[student_valid[j]].size() - 30; k < students_all[student_valid[j]].size(); k++){
					nose_range_x.push_back(students_all[student_valid[j]][k].loc.x);
					nose_range_y.push_back(students_all[student_valid[j]][k].loc.y);
				}

				float max_nose_x = *max_element(nose_range_x.begin(), nose_range_x.end());
				float min_nose_x = *min_element(nose_range_x.begin(), nose_range_x.end());
				float max_nose_y = *max_element(nose_range_y.begin(), nose_range_y.end());
				float min_nose_y = *min_element(nose_range_y.begin(), nose_range_y.end());
				if (abs(nose_range_x[nose_range_x.size() - 1] - max_nose_x) < thre && abs(nose_range_x[nose_range_x.size() - 1] - min_nose_x) < thre && abs(nose_range_y[nose_range_y.size() - 1] - max_nose_y) < thre && abs(nose_range_y[nose_range_y.size() - 1] - min_nose_y) < thre){
					students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze = true;
				}
			}
		}

		//----------------�������--------------------------
		if (class_info.all_disscussion == true){
			cv::putText(image, status3, cv::Point2f(image.size[1] / 2, image.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 255), 1);
		}
		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion == true){
			cv::putText(image, status3, cv::Point2f(x1, y1), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 255), 1);
		}

		//----------------�������----------------------------
		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze == true){
			cv::putText(image, status4, cv::Point2f(x1 + 10, y1 + 10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(209, 206, 0), 1);
		}
		//---------------�������-----------------------------
		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing == true){
			cv::putText(image, status2, cv::Point2f(x1, y1), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 102, 224), 1);
		}
		//----------------�����ͷ----------------------------
		if (class_info.all_bow_head == true){
			cv::putText(image, status5, cv::Point2f(image.size[1] / 2, image.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 255), 1);
		}
		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head == true){
			cv::putText(image, status5, cv::Point2f(x1, y1), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 191, 0), 1);
		}
		//-----------------�������-----------------------------
		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].raising_hand == true){
			cv::putText(image, status1, cv::Point2f(x1, y1), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 139), 1);
		}
	}
	char buff[100];
	if (num_of_back >= 10){
		int disscuss_people = 2 * num_of_back;
		sprintf(buff, "4-students'discussion: %d", disscuss_people);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
		sprintf(buff, "2-students'discussion: %d", 0);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	else if (num_of_disscuss >= 10){
		int disscuss_people = num_of_disscuss;
		sprintf(buff, "4-students'discussion: %d", 0);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
		sprintf(buff, "2-students'discussion: %d", disscuss_people);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	else{
		int disscuss_front = num_of_disscuss;
		int disscuss_back = 2 * num_of_back;
		sprintf(buff, "4-students'discussion: %d", disscuss_back);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
		sprintf(buff, "2-students'discussion: %d", disscuss_front);
		cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
}