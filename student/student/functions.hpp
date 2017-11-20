#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
void refine(Rect& bbox, cv::Mat& img);
float PointToLineDis(Point2f cur, Point2f start, Point2f end);
bool PtInAnyRect1(Point2f pCur, Rect search);
float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3);
int cosDistance(const cv::Mat q, const cv::Mat r, float& distance);
float euDistance(Point2f q, Point2f r);
int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance);
float Compute_IOU(const cv::Rect& rectA, const cv::Rect& rectB);
bool greate2(vector<float>a, vector<float>b);
#endif