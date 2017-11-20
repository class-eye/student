#ifndef _BEHAVIOR_H_
#define _BEHAVIOR_H_
#include <opencv2/core/core.hpp>
#include "student.hpp"
using namespace cv;
void Analys_Behavior(vector<vector<Student_Info>>students_all, vector<int>&student_valid, Class_Info &class_info, Mat &image);
#endif