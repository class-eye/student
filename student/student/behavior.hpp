#ifndef _BEHAVIOR_H_
#define _BEHAVIOR_H_
#include <opencv2/core/core.hpp>
#include "student.hpp"
using namespace cv;
void Analys_Behavior(vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info, Mat &image, int &n, PLAYM4_SYSTEM_TIME &pstSystemTime,int &num_turn_body);
#endif