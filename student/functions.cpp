#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  
#include "student/student.hpp"
#include "student/functions.hpp"
#include<cmath>
#ifdef __unix__
#include <json/json.h>
//#include <python2.7/Python.h>
#endif
using namespace cv;
using namespace std;

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
float PointToLineDis(Point2f cur, Point2f start, Point2f end){
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
float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float x_1 = x2 - x1;
	float x_2 = x3 - x2;
	float y_1 = y2 - y1;
	float y_2 = y3 - y2;
	float lx = sqrt(x_1*x_1 + y_1*y_1);
	float ly = sqrt(x_2*x_2 + y_2*y_2);
	return 180.0 - acos((x_1*x_2 + y_1*y_2) / lx / ly) * 180 / 3.1415926;
}
int cosDistance(const cv::Mat q, const cv::Mat r, float& distance)
{
	assert((q.rows == r.rows) && (q.cols == r.cols));
	float fenzi = q.dot(r);
	float fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
	distance = fenzi / fenmu;
	return 0;
}
float euDistance(Point2f q, Point2f r){
	float distance = sqrt(pow(q.x - r.x, 2) + pow(q.y - r.y, 2));
	return distance;
}
int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance)
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
bool greate2(vector<float>a, vector<float>b){
	return a[1] > b[1];
}
bool greate3(Student_Info a, Student_Info b){
	return a.energy > b.energy;
}

//Json::Value root_all;
void writeJson(vector<int>&student_valid, vector<vector<Student_Info>>&students_all, vector<Class_Info>&class_info_all, string &output,int &n){
	Json::Value root;
	vector<string>human;
	for (int i = 0; i < student_valid.size(); i++){
		string human_x = "student" + to_string(student_valid[i]);
		human.push_back(human_x);
	}
	//root["Frame"] = n;
	Json::Value class_infomation;
	for (int i = 0; i < class_info_all.size(); i++){
		if (class_info_all[i].all_bow_head == true){
			class_infomation["all_bow_head"].append(class_info_all[i].cur_frame);
		}
		if (class_info_all[i].all_disscussion_2 == true){
			class_infomation["2-students'disscussion"].append(class_info_all[i].cur_frame);
		}
		if (class_info_all[i].all_disscussion_4 == true){
			class_infomation["4-students'disscussion"].append(class_info_all[i].cur_frame);
		}
	}
	root["class_infomation"] = Json::Value(class_infomation);

	for (int i = 0; i < student_valid.size(); i++){

		Json::Value behavior_infomation;
		behavior_infomation["max_energy"] = students_all[student_valid[i]][0].cur_frame1;
		for (int j = 1; j < students_all[student_valid[i]].size(); j++){
			if (students_all[student_valid[i]][j].bow_head == true){
				behavior_infomation["bow_head"].append(students_all[student_valid[i]][j].cur_frame1);
			}
			if (students_all[student_valid[i]][j].daze == true){
				behavior_infomation["daze"].append(students_all[student_valid[i]][j].cur_frame1);
			}
			if (students_all[student_valid[i]][j].raising_hand == true){
				behavior_infomation["rasing_hand"].append(students_all[student_valid[i]][j].cur_frame1);
			}
			if (students_all[student_valid[i]][j].standing == true){
				behavior_infomation["standing"].append(students_all[student_valid[i]][j].cur_frame1);
			}
		}
		root[human[i]] = Json::Value(behavior_infomation);

		/*if (students_all[student_valid[i]].size() == 1)continue;
		else{
			Json::Value student_loc;
			for (int j = 0; j < 8; j++){
				Json::Value part_loc;
				int x = students_all[student_valid[i]][students_all[student_valid[i]].size() - 1].all_points[j].x;
				int y = students_all[student_valid[i]][students_all[student_valid[i]].size() - 1].all_points[j].y;
				part_loc.append(x);
				part_loc.append(y);
				student_loc["Parts'location"].append(part_loc);
			}
			root[human[i]] = Json::Value(student_loc);
		}*/
	}
	//root_all.append(root);
	int pos1 = output.find_last_of("/");
	int pos2 = output.find_last_of(".");
	string videoname = output.substr(pos1, pos2 - pos1);
	ofstream out;
	string jsonfile = output.substr(0,pos1)+"/" + videoname + ".json";
	out.open(jsonfile);
	Json::StyledWriter sw;
	out << sw.write(root);
	out.close();

}
void drawGrid(Mat &image, vector<int>student_valid,vector<vector<Student_Info>>students_all){
	vector<vector<int>>orderr = { { 14, 3, 21, 19, 5, 17 }, { 15, 20, 2, 9, 12, 22, 24 }, { 7, 0, 1, 26, 50, 48 }, { 27, 4, 6, 32, 29, 49, 13 }, { 23, 41, 8, 34, 39, 35, 31 }, { 38, 43, 28, 10, 16, 42, 51 }, { 18, 37, 33, 30, 44, 40 }, {47,45,25,11,46,36} };
	for (int i = 0; i < orderr.size(); i++){
		for (int j = 0; j < orderr[i].size()-1; j++){
			auto iter1 = find(student_valid.begin(), student_valid.end(), orderr[i][j]);
			int index1 = distance(student_valid.begin(), iter1);
			auto iter2 = find(student_valid.begin(), student_valid.end(), orderr[i][j+1]);
			int index2 = distance(student_valid.begin(), iter2);
			/*Point2f start = students_all[orderr[i][j]][students_all[orderr[i][j]].size() - 1].loc;
			Point2f end = students_all[orderr[i][j+1]][students_all[orderr[i][j+1]].size() - 1].loc;*/
			Point2f start = students_all[student_valid[index1]][students_all[student_valid[index1]].size() - 1].loc;
			Point2f end = students_all[student_valid[index2]][students_all[student_valid[index2]].size() - 1].loc;
			line(image, start, end, Scalar(255, 0, 0), 2, 8, 0);
		}
	}
}