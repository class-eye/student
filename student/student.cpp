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

static int euDistance(Point2f q, Point2f r, float &distance){
	distance = sqrt(pow(q.x - r.x, 2) + pow(q.y - r.y, 2));
	return 0;
}

static int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance)
{
	cv::Mat q(query_feature);
	cv::Mat r(ref_feature);
	cosDistance(q, r, distance);   //cos distance
	return 0;
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

int standard_frame = 25;
vector<int>student_valid;


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
			for (int i = 0; i < pose.subset.size(); i++){
				int symbol_raise = 0;
				int symbol_stand1 = 0;
				int symbol_stand2 = 0;
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
					/*if (pose.subset[i][19] >= 1 && score >= 0.1){
						if (pose.subset[i][1] != -1){
						Point2f nose_loc;
						nose_loc.x = pose.candicate[pose.subset[i][1]][0];
						nose_loc.y = pose.candicate[pose.subset[i][1]][1];
						all_nose.push_back(nose_loc);
						}
						for (int j = 0; j < 2; j++){
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
						}*/

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
					if (student_info.raising_hand || student_info.standing){
						student_have_action.push_back(student_info);
					}

				}//if (pose.subset[i][19] >= 3 && score >= 0.4) end
				for (int j = 0; j < 8; j++){
					if (!(x[j] || y[j])){
						continue;
					}
					else{
						cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
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
			//	float dis = PointToLineDis(all_nose[i],start,end);
			//	vector<float>tmp_dis;
			//	tmp_dis.push_back(i);
			//	tmp_dis.push_back(dis);
			//	all_dis.push_back(tmp_dis);
			//	
			//}
			//cout << all_dis.size() << endl;
			//sort(all_dis.begin(), all_dis.end(), dis_smaller);
			//for (int i = 0; i < classroom_row; i++){	
			//	int thre = (all_dis[i*classroom_col + 1][1] + all_dis[i*classroom_col + 2][1] + all_dis[i*classroom_col + 3][1] + all_dis[i*classroom_col + 4][1]) / 4;
			//	row_threshold[i]=thre;
			//	cout << thre << endl;
			//}
			//
			//for (int i = 0; i < all_nose.size(); i++){
			//	for (int j = 0; j < row_threshold.size(); j++){
			//		
			//		if (all_nose[all_dis[i][0]].y > (image.size().height / 2)){	
			//			
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 50){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//		
			//			
			//		}
			//		
			//		else if (all_nose[all_dis[i][0]].y < (image.size().height / 2) && all_nose[all_dis[i][0]].y >(image.size().height / 3)){
			//		
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 30){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//			
			//		}
			//		
			//		else{
			//			if (abs(all_dis[i][1] - row_threshold[j]) < 15){
			//				all_row[j].push_back(all_nose[all_dis[i][0]]);
			//				break;
			//			}
			//		}
			//	}
			//}
			//
			//for (int i = 0; i < classroom_row; i++){
			//	/*for (int j = 0; j < classroom_col; j++){		
			//		all_row[i].push_back(all_nose[i*classroom_col+j]);
			//	}*/
			//	sort(all_row[i].begin(), all_row[i].end(), x_smaller);
			//}
			//
			//for (int i = 0; i < classroom_row; i++){
			//	for (int j = 0; j < all_row[i].size()-1; j++){
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

		}
		string outputp1 = "../output/" + to_string(n) + ".jpg";
		cv::imwrite(outputp1, image);
		timer.Toc();
		cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;


	}   //if (n % standard_frame == 0) end
	return student_have_action;
}
