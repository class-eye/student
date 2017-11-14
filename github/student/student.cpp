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
using namespace cv;
using namespace std;
using namespace caffe;

void refine(Rect& bbox, cv::Mat& img)
{
	if (bbox.x < 0 && bbox.y < 0 && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float a = bbox.x;
		float  b = bbox.y;
		bbox.x = 0;
		bbox.y = 0;
		bbox.width = bbox.width + a;
		bbox.height = bbox.height + b;
	}
	if (bbox.x < 0 && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width<img.size[1] && bbox.y + bbox.height>img.size[0]){
		float  a = bbox.x;
		bbox.x = 0;
		bbox.width = bbox.width + a;
		bbox.height = img.size[0] - bbox.y;
	}
	if (0 < bbox.x < img.size[1] && bbox.y<0 && bbox.x + bbox.width>img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.y;
		bbox.y = 0;
		bbox.width = img.size[1] - bbox.x;
		bbox.height = bbox.height + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y<img.size[0] && bbox.x + bbox.width>img.size[1] && bbox.y + bbox.height > img.size[0]){
		bbox.width = img.size[1] - bbox.x;
		bbox.height = img.size[0] - bbox.y;
	}
	if (bbox.x < 0 && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.x;
		bbox.x = 0;
		bbox.width = bbox.width + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y<img.size[0] && bbox.x + bbox.width>img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		bbox.width = img.size[1] - bbox.x;
	}
	if (0 < bbox.x < img.size[1] && bbox.y < 0 && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.y;
		bbox.y = 0;
		bbox.height = bbox.height + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width<img.size[1] && bbox.y + bbox.height>img.size[0]){
		bbox.height = img.size[0] - bbox.y;
	}

}
float PointToLineDis(Point2f cur,Point2f start,Point2f end){
	double a, b, c, dis;
	a = end.y - start.y;
	b = start.x - end.x;
	c = end.x * start.y - start.x * end.y;
	dis = abs(a * cur.x + b * cur.y + c) / sqrt(a * a + b * b);
	return dis;
}
bool PtInAnyRect1(Point2f pCur, Rect search)
{
	Point2f pLT, pRT, pLB, pRB;
	pLT.x = search.x;
	pLT.y = search.y;
	pRT.x = search.x + search.width;
	pRT.y = search.y;
	pLB.x = search.x;
	pLB.y = search.y + search.height;
	pRB.x = search.x + search.width;
	pRB.y = search.y + search.height;
	//任意四边形有4个顶点
	std::vector<double> jointPoint2fx;
	std::vector<double> jointPoint2fy;
	int nCount = 4;
	Point2f RectPoint2fs[4] = { pLT, pLB, pRB, pRT };
	int nCross = 0;
	for (int i = 0; i < nCount; i++)
	{
		Point2f pStart = RectPoint2fs[i];
		Point2f pEnd = RectPoint2fs[(i + 1) % nCount];

		if (pCur.y < min(pStart.y, pEnd.y) || pCur.y > max(pStart.y, pEnd.y))
			continue;

		double x = (double)(pCur.y - pStart.y) * (double)(pEnd.x - pStart.x) / (double)(pEnd.y - pStart.y) + pStart.x;
		if (x > pCur.x)nCross++;
	}
	return (nCross % 2 == 1);

}
static float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float x_1 = x2 - x1;
	float x_2 = x3 - x2;
	float y_1 = y2 - y1;
	float y_2 = y3 - y2;
	float lx = sqrt(x_1*x_1 + y_1*y_1);
	float ly = sqrt(x_2*x_2 + y_2*y_2);
	return 180.0 - acos((x_1*x_2 + y_1*y_2) / lx / ly) * 180 / 3.1415926;
}
static int cosDistance(const cv::Mat q, const cv::Mat r, float& distance)
{
	assert((q.rows == r.rows) && (q.cols == r.cols));
	float fenzi = q.dot(r);
	float fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
	distance = fenzi / fenmu;
	return 0;
}
static float euDistance(Point2f q, Point2f r){
	float distance = sqrt(pow(q.x - r.x, 2) + pow(q.y - r.y, 2));
	return distance;
}
static int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance)
{
	cv::Mat q(query_feature);
	cv::Mat r(ref_feature);
	cosDistance(q, r, distance);   //cos distance
	return 0;
}
float Compute_IOU(const cv::Rect& rectA, const cv::Rect& rectB){
	if (rectA.x > rectB.x + rectB.width) { return 0.; }
	if (rectA.y > rectB.y + rectB.height) { return 0.; }
	if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
	if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
	float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
	float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
	float intersection = colInt * rowInt;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float intersectionPercent = intersection / (areaA + areaB - intersection);
	/*intersectRect.x = max(rectA.x, rectB.x);
	intersectRect.y = max(rectA.y, rectB.y);
	intersectRect.width = min(rectA.x + rectA.width, rectB.x + rectB.width) - intersectRect.x;
	intersectRect.height = min(rectA.y + rectA.height, rectB.y + rectB.height) - intersectRect.y;*/
	return intersectionPercent;
}
vector<int>student_valid;
vector<vector<Student_Info>>students_all(70);
int standard_frame = 25;
int max_student_num = 0;

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
						standard_rect.height = wid1 + wid2;
						refine(standard_rect, frame);
						//cv::rectangle(frame, standard_rect, Scalar(0, 0, 255), 2, 8, 0);
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
						
						student_ori.output_body_dir = output_body[i];

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


inline bool y_bigger(Point2f a, Point2f b){
	return a.y > b.y;
}
inline bool x_smaller(Point2f a, Point2f b){
	return a.x < b.x;
}
inline bool dis_smaller(vector<float>a, vector<float>b){
	return a[1] < b[1];
}
inline bool greate2(vector<float>a, vector<float>b){
	return a[1] > b[1];
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
			cout << "pose detect cost " << timer.Elasped() / 1000.0 << " s" << endl;
			timer.Tic();*/
			int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };

			int x[18];
			int y[18];
			string status1 = "raising hand";
			string status2 = "standing";

			//----------------get the location of student--------------------

			/*Mat img_trans = imread("../0.jpg");
			Mat mask = imread("../0.jpg", 0);*/
			//line(img_trans, Point2f(img_width, 0), Point2f(img_width, img_height), cv::Scalar(0, 0, 0), 1, 8, 0);


			//------------------raising or standing--------------------------
			vector<Point2f>all_nose;
			for (int i = 0; i < pose.subset.size(); i++){
				Student_Info student_info;
				int symbol_raise = 0;
				int symbol_stand1 = 0;
				int symbol_stand2 = 0;
				int v = 0;
				float score = float(pose.subset[i][18]) / pose.subset[i][19];
				if (pose.subset[i][19] >= 3 && score >= 0.4){
					if (pose.subset[i][1] != -1){
						Point2f nose_loc;
						nose_loc.x = pose.candicate[pose.subset[i][1]][0];
						nose_loc.y = pose.candicate[pose.subset[i][1]][1];
						all_nose.push_back(nose_loc);
					}
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

					//-----------判断站立--------------------

					if (v == 0){
						float angle_r = CalculateVectorAngle(x[2], y[2], x[3], y[3], x[4], y[4]);
						float angle_l = CalculateVectorAngle(x[5], y[5], x[6], y[6], x[7], y[7]);
						if ((y[4] > y[3] && y[3] > y[2]) && (y[7] > y[6] && y[6] > y[5])){
							float longer_limb = max(abs(y[4] - y[2]), abs(y[7] - y[5]));
							float shorter_limb = min(abs(y[4] - y[2]), abs(y[7] - y[5]));
							float longer_width = max(abs(x[5] - x[2]), abs(x[7] - x[4]));
							float shorter_width = min(abs(x[5] - x[2]), abs(x[7] - x[4]));
							if (shorter_limb / longer_limb > 0.75 && shorter_width / longer_width > 0.7){
								bool Vertical_l = false;
								bool Vertical_r = false;
								if (abs(y[4] - y[3]) >= abs(x[4] - x[3]) && abs(y[2] - y[3]) >= abs(x[2] - x[3]))Vertical_r = true;
								if (abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5]))Vertical_l = true;
								if ((y[4] - y[3]) > 10 && (y[7] - y[6]) > 10 && float(y[4] - y[3]) / float(y[3] - y[2]) >= 0.7  && float(y[7] - y[6]) / float(y[6] - y[5]) >= 0.7 && Vertical_r &&Vertical_l && (angle_r > 135 && angle_l > 135)){
									symbol_stand1 = 1;
								}
								/*if ((y[7] - y[6] > 10) && float(y[7] - y[6]) / float(y[6] - y[5]) >= 0.7 &&Vertical_l && (angle_l > 135 && angle_r > 120)){
								symbol_stand2 = 1;
								}*/
							}
						}
					}

					//----------------判断举手-----------------------------

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

					//----------------如果举手，如果站立--------------------

					if (symbol_raise == 1){    //如果举手
						for (int j = 0; j < 8; j++){
							if (!(x[j] || y[j])){
								continue;
							}
							else{
								cv::circle(image, Point2f(x[j], y[j]), 5, cv::Scalar(255, 255, 255), -1);
							}
						}
						student_info.loc.x = x[0];
						student_info.loc.y = y[0];
						student_info.raising_hand = true;
						cv::putText(image, status1, cv::Point2f(x[1], y[1]), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255), 1);
					}
					else student_info.raising_hand = false;
					if (symbol_stand1 || symbol_stand2){        //如果站立
						for (int j = 0; j < 8; j++){
							if (!(x[j] || y[j])){
								continue;
							}
							else{
								cv::circle(image, Point2f(x[j], y[j]), 5, cv::Scalar(255, 255, 255), -1);
							}
						}
						//Point2fs[i].y = (y[6] + y[7]) / 2.0 + 15;
						student_info.loc.x = x[0];
						student_info.loc.y = y[0];
						cv::putText(image, status2, cv::Point2f(x[1], y[1]), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255), 1);
						student_info.standing = true;
					}
					else student_info.standing = false;
					if (student_info.raising_hand || student_info.standing){               //-----------------------------------------------------------------------------------------------
						student_have_action.push_back(student_info);
					}



					//--------------------use IOU to classify-------------------------------

					//----------------obtain a rect range for i person in a new frame---------------------
					if (/*pose.subset[i][0] != -1&&*/pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){

						float wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						float wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						float wid = MAX(wid1, wid2);
						if (wid == 0)continue;

						Rect cur_rect;
						if (symbol_stand1 || symbol_stand2){
							cur_rect.x = pose.candicate[pose.subset[i][1]][0]-wid;
							cur_rect.y = pose.candicate[pose.subset[i][1]][1];
							cur_rect.width = wid1 + wid2;
							cur_rect.height = wid1 + wid2 + 60;
						}
						else{
							cur_rect.x = pose.candicate[pose.subset[i][1]][0]-wid;
							cur_rect.y = pose.candicate[pose.subset[i][1]][1];
							cur_rect.width = wid1 + wid2;
							cur_rect.height = wid1 + wid2;
						}
						refine(cur_rect, image);

						//cv::rectangle(image, cur_rect, Scalar(0, 255, 0), 2, 8, 0);
						student_info.body_bbox = cur_rect;

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
				
					
					for (int j = 0; j < 2; j++){
						if (!(x[j] || y[j])){
							continue;
						}
						else{
							cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
						}
					}
				} //if (pose.subset[i][19] >= 3 && score >= 0.4) end	
			}//for (int i = 0; i < pose.subset.size(); i++) end
		
			if (students_all[student_valid[2]].size() <= 6){
				for (int j = 0; j < student_valid.size(); j++){
					for (int k = 1; k < students_all[student_valid[j]].size() - 1; k++){
						if (students_all[student_valid[j]][k].front == true && students_all[student_valid[j]][k+1].front == true){
							line(image, Point2f(students_all[student_valid[j]][k].loc.x, students_all[student_valid[j]][k].loc.y), Point2f(students_all[student_valid[j]][k + 1].loc.x, students_all[student_valid[j]][k + 1].loc.y), cv::Scalar(0, 0, 255), 2, 8, 0);
						}
						else{
							line(image, Point2f(students_all[student_valid[j]][k].neck_loc.x, students_all[student_valid[j]][k].neck_loc.y), Point2f(students_all[student_valid[j]][k + 1].neck_loc.x, students_all[student_valid[j]][k + 1].neck_loc.y), cv::Scalar(255, 0, 0), 2, 8, 0);
						}
					}
				}
			}
			else{
				for (int j = 0; j < student_valid.size(); j++){
					int k1 = students_all[student_valid[j]].size() - 5;
					int k2 = students_all[student_valid[j]].size() - 4;
					int k3 = students_all[student_valid[j]].size() - 3;
					int k4 = students_all[student_valid[j]].size() - 2;
					int k5 = students_all[student_valid[j]].size() - 1;
					if (students_all[student_valid[j]][k1].front == true && students_all[student_valid[j]][k2].front == true && students_all[student_valid[j]][k3].front == true && students_all[student_valid[j]][k4].front == true && students_all[student_valid[j]][k5].front == true)
					{
						for (int k = students_all[student_valid[j]].size() - 6; k < students_all[student_valid[j]].size() - 1; k++){
							line(image, Point2f(students_all[student_valid[j]][k].loc.x, students_all[student_valid[j]][k].loc.y), Point2f(students_all[student_valid[j]][k + 1].loc.x, students_all[student_valid[j]][k + 1].loc.y), cv::Scalar(0, 0, 255), 2, 8, 0);
						}

					}
					else{
						for (int k = students_all[student_valid[j]].size() - 6; k < students_all[student_valid[j]].size() - 1; k++){
							line(image, Point2f(students_all[student_valid[j]][k].neck_loc.x, students_all[student_valid[j]][k].neck_loc.y), Point2f(students_all[student_valid[j]][k + 1].neck_loc.x, students_all[student_valid[j]][k + 1].neck_loc.y), cv::Scalar(255, 0, 0), 2, 8, 0);
						}
					}
				}
			}
			
			//----------------画网格-----------------------	

			//cv::putText(image, to_string(all_nose.size()), cv::Point2f(image.size[0] / 2, image.size[1] / 2), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255, 255, 255), 1);
			//Point2f start(5, 720);
			//Point2f end(1256, 720);
			//int classroom_col = 8;
			//int classroom_row = (all_nose.size()) / classroom_col;
			//vector<int>row_threshold(classroom_row);
			//vector<vector<Point2f>>all_row(classroom_row);
			//vector<vector<float>>all_dis;
			////sort(all_nose.begin(), all_nose.end(), y_bigger);
			//for (int i = 0; i < all_nose.size(); i++){
			//	float dis = PointToLineDis(all_nose[i], start, end);
			//	vector<float>tmp_dis;
			//	tmp_dis.push_back(i);
			//	tmp_dis.push_back(dis);
			//	all_dis.push_back(tmp_dis);
			//}
			//cout << all_dis.size() << endl;
			//sort(all_dis.begin(), all_dis.end(), dis_smaller);
			//for (int i = 0; i < classroom_row; i++){
			//	int thre = (all_dis[i*classroom_col + 1][1] + all_dis[i*classroom_col + 2][1] + all_dis[i*classroom_col + 3][1] + all_dis[i*classroom_col + 4][1]) / 4;
			//	row_threshold[i] = thre;
			//	cout << thre << endl;
			//}
			//for (int i = 0; i < all_nose.size(); i++){
			//	for (int j = 0; j < row_threshold.size(); j++){
			//		if (all_nose[all_dis[i][0]].y > (image.size().height / 2)){
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 50){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//		}
			//		else if (all_nose[all_dis[i][0]].y < (image.size().height / 2) && all_nose[all_dis[i][0]].y >(image.size().height / 3)){
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 35){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//		}
			//		else{
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 20){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//		}
			//	}
			//}
			//for (int i = 0; i < classroom_row; i++){
			//	/*for (int j = 0; j < classroom_col; j++){
			//		all_row[i].push_back(all_nose[i*classroom_col+j]);
			//		}*/
			//	sort(all_row[i].begin(), all_row[i].end(), x_smaller);
			//}
			//for (int i = 0; i < classroom_row; i++){
			//	for (int j = 0; j < all_row[i].size() - 1; j++){
			//		/*float bili = abs(all_row[i][j].x - all_row[i][j + 1].x) / abs(all_row[i][j].y - all_row[i][j + 1].y);
			//		if (bili>4 && all_row[i][j + 1].x != 0 && all_row[i][j].x != 0 && all_row[i][j + 1].y != 0 && all_row[i][j].y != 0)*/
			//		line(image, Point2f(all_row[i][j].x, all_row[i][j].y), Point2f(all_row[i][j + 1].x, all_row[i][j + 1].y), cv::Scalar(255, 0, 0), 2, 8, 0);
			//	}
			//}
			//for (int i = 0; i < classroom_col; i++){
			//	for (int j = 0; j < classroom_row - 1; j++){
			//		line(image, Point2f(all_row[j][i].x, all_row[j][i].y), Point2f(all_row[j + 1][i].x, all_row[j + 1][i].y), cv::Scalar(0, 255, 0), 2, 8, 0);
			//	}
			//}


			string outputp1 = "../output/" + to_string(n) + ".jpg";
			cv::imwrite(outputp1, image);
			timer.Toc();
			cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;
		} //if (n % standard_frame == 0) end
	return student_have_action;
}
