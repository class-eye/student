#include <iostream>
#include <string>
#include <cstring> 
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  
#include <caffe/caffe.hpp>
#include <thread>
#include "student/student.hpp"
using namespace std;
using namespace cv;
using namespace caffe;
using namespace fs;

vector<string>output_body;
int main(){
	if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 1);
	}
	Net net1("../models/pose_deploy.prototxt");
	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
	
	/*Net net2("../models/feat.prototxt");
	net2.CopyTrainedLayersFrom("../models/feat.caffemodel");*/

	//for (int i = 0; i <70; i++){
	//	string output = "../bodydir/body" + to_string(i);
	//	/*if (!fs::IsExists(output)){
	//	fs::MakeDir(output);
	//	}*/
	//	output_body.push_back(output);
	//}

	VideoCapture capture("../taolun1.mp4");
	if (!capture.isOpened())
	{
		printf("video loading fail");
	}
	Mat frame;
	
	int n = 0;
	vector<Student_Info>student_info;
	int frameToStart = 6750;    //taolun1
	//int frameToStart = 800;       //taolun2	
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	while (true)
	{
		if (!capture.read(frame)){
			break;
		}
		
		if (n<25*10){
			cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
			PoseInfo pose1;
			GetStandaredFeats(net1, pose1,frame,n);		
		}	

		else{
			cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
			PoseInfo pose;
			student_info = student_detect(net1, frame, n, pose);
		}
		n++;
	}


	//--------------use pic to test----------------------------
	
	/*vector<string>imagelist = fs::ListDir("/home/lw/student_api/inputimg", { "jpeg" });
	int n = 0;
	for (int i = 0; i < imagelist.size(); i++){
		string imagep = "/home/lw/student_api/inputimg/" + imagelist[i];
		Mat image = imread(imagep);
		cv::resize(image, image, Size(0, 0), 2 / 3., 2 / 3.);
		PoseInfo pose;
		vector<Student_Info>student_info;
		student_info = student_detect(net1, image, n, pose);
		n++;
	}*/

}

