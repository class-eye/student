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
#include "student/functions.hpp"
#include "incCn/HCNetSDK.h"  
#include "incCn/PlayM4.h" 
#include<tuple>
using namespace std;
using namespace cv;
using namespace caffe;
using namespace fs;

vector<Class_Info>class_info_all;
vector<int>student_valid;
vector<vector<Student_Info>>students_all(70);
int n = 0;
int max_student_num = 0;
std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
string output;

void initValue(int &n, int &max_student_num, vector<Class_Info>&class_info_all, vector<int>&student_valid, vector<vector<Student_Info>>&students_all){
	n = 0;
	max_student_num = 0;
	student_valid.clear();
	for (int i = 0; i < 70; i++){
		students_all[i].clear();
	}
	class_info_all.clear();
}

void yv12toYUV(char *outYuv, char *inYv12, int width, int height, int widthStep)
{
	int col, row;
	unsigned int Y, U, V;
	int tmp;
	int idx;
	for (row = 0; row<height; row++)
	{
		idx = row * widthStep;
		int rowptr = row*width;

		for (col = 0; col<width; col++)
		{
			tmp = (row / 2)*(width / 2) + (col / 2);
			Y = (unsigned int)inYv12[row*width + col];
			U = (unsigned int)inYv12[width*height + width*height / 4 + tmp];
			V = (unsigned int)inYv12[width*height + tmp];
			outYuv[idx + col * 3] = Y;
			outYuv[idx + col * 3 + 1] = U;
			outYuv[idx + col * 3 + 2] = V;
		}
	}
}

#define READ_BUF_SIZE 200
#define HIK_HEAD_LEN 40

Net *a;
Net *b;
void CALLBACK DecCBFun(int nPort, char * pBuf, int nSize, FRAME_INFO * pFrameInfo, void* nReserved1, int nReserved2)
{
	{
		if (caffe::GPUAvailable()) {
			caffe::SetMode(caffe::GPU, 0);
		}
		//caffe::SetMode(caffe::CPU, -1);
		long lFrameType = pFrameInfo->nType;
		int frameH = pFrameInfo->nHeight;
		int frameW = pFrameInfo->nWidth;
		if (lFrameType == T_YV12)
		{
			PLAYM4_SYSTEM_TIME pstSystemTime;
			PlayM4_GetSystemTime(nPort, &pstSystemTime);
			//cout << "tmptime--" << pstSystemTime.dwYear << " " << pstSystemTime.dwMon << " " << pstSystemTime.dwDay << " " << pstSystemTime.dwHour << " " << pstSystemTime.dwMin << " " << pstSystemTime.dwSec << endl;
			IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//得到图像的Y分量    
			yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//得到全部RGB图像  
			IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);
			cvCvtColor(pImgYCrCb, pImg, CV_YCrCb2RGB);
			cv::Mat img(pImg, true);
			cvReleaseImage(&pImgYCrCb);
			cvReleaseImage(&pImg);
			cv::resize(img, img, Size(0, 0), 2 / 3., 2 / 3.);
			if (n < 20 /*&& n>14 */){
				PoseInfo pose1;
				GetStandaredFeats(*a, pose1, img, n, output, max_student_num, students_all, student_valid,  class_info_all);
			}
			else {
				PoseInfo pose;
				student_info = student_detect(*a, *b, img, n, pose, output, pstSystemTime,students_all,student_valid,class_info_all);
				/*vector<vector<Student_Info>>students_all = get<0>(student_info);
				vector<Class_Info>class_info_all = get<1>(student_info);*/
			}
			n++;

		}
	}
	MemPoolClear();
}

int main()
{
	bool bFlag = false;
	int  nError = 0;
	FILE* fp = NULL;
	unsigned char* pBuffer = NULL;
	int g_nPort = -1;

	//caffe::SetMode(caffe::CPU, -1);

	if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 0);	
	}
	Net net1("../models/pose_deploy.prototxt");
	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
	a = &net1;	
	Net net2("../models/handsnet.prototxt");
	net2.CopyTrainedLayersFrom("../models/handsnet_iter_12000.caffemodel");
	b = &net2;
	//string videodir = "/home/lw/student_api/inputvideo";
	//string videodir = "/home/data/jiangbo/xiaoxue/code1/54";
	string resultdir = "/home/lw/student_api/output";

	string videodir = "/home/lw/data/code1/54";
	//string resultdir = "/home/lw/data/hands";

	if (!fs::IsExists(resultdir)){
		fs::MakeDir(resultdir);
	}
	
	//int i=0;  //53
	//int i = 5;   //54
	//int i=0;   //62
	int i = 0;  //63
	vector<string>videolist = fs::ListDir(videodir, { "mp4" });
	while (i < 1){
		initValue(n, max_student_num, class_info_all, student_valid, students_all);
		//string videoname = videolist[i];
		string videoname = "54_english_0916_6_1.mp4";
		output = resultdir + "/" + videoname;
		if (!fs::IsExists(output)){
			fs::MakeDir(output);
		}
		cout << videolist.size()<<"  "<<videoname << endl;
		string videopath = videodir + "/" + videoname;
		//获取播放库通道号
		PlayM4_GetPort(&g_nPort);
		
		
		cout << "###duankouhao###::" << g_nPort << endl;
		fp = fopen(videopath.c_str(), ("rb"));   //######
		

		if (fp == NULL)
		{
			printf("cannot open the file !\n");
			return 0;
		}
		pBuffer = new unsigned char[READ_BUF_SIZE];
		if (pBuffer == NULL)
		{
			return 0;
		}
		//读取文件中海康文件头
		fread(pBuffer, HIK_HEAD_LEN, 1, fp);

		


		//设置流模式类型 
		PlayM4_SetStreamOpenMode(g_nPort, STREAME_FILE);
		//打开流模式
		PlayM4_OpenStream(g_nPort, pBuffer, HIK_HEAD_LEN, 1024 * 1024);
		//设置解码回调
		PlayM4_SetDecCallBackMend(g_nPort, DecCBFun, NULL);
		PlayM4_SetDecodeFrameType(g_nPort, 1);
		PlayM4_Play(g_nPort, NULL);
	
		/*cout << "video time: " << PlayM4_GetFileTime(g_nPort) << endl;
		cout << "video frames: " << PlayM4_GetFileTotalFrames(g_nPort) << endl;*/
		//bool a=PlayM4_SetCurrentFrameNum(g_nPort, 300);  //从第几帧开始
		
		while (!feof(fp))
		{
			fread(pBuffer, READ_BUF_SIZE, 1, fp);
			while (1)
			{
				bFlag = PlayM4_InputData(g_nPort, pBuffer, READ_BUF_SIZE);
				if (bFlag == false)
				{
					nError = PlayM4_GetLastError(g_nPort);

					//若缓冲区满，则重复送入数据
					if (nError == PLAYM4_BUF_OVER)
					{
						sleep(2);
						continue;
					}
				}
				//若送入成功，则继续读取数据送入到播放库缓冲
				break;
			}

		}
		//---------------------------------------
		cout << "**** stop" << endl;
		//PlayM4_Stop(g_nPort);
		
		//关闭流，回收源数据缓冲
		PlayM4_CloseStream(g_nPort);
		cout << "***close" << endl;
		//释放播放库端口号
		//PlayM4_FreePort(g_nPort);
		cout << "***1" << endl;

		i++;
		if (fp != NULL)
		{
			fclose(fp);
			fp = NULL;
		}
		if (pBuffer != NULL)
		{
			delete[] pBuffer;
			pBuffer = NULL;
		}
		
	}
	return 0;
}

//-------------------------------------------------OpenCV------------------------------------------------------

//int main(){
//
//	if (caffe::GPUAvailable()) {
//		caffe::SetMode(caffe::GPU, 1);
//	}
//	Net net1("../models/pose_deploy.prototxt");
//	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
//	//string videodir = "/home/data/jiangbo/xiaoxue/arranged/qian/1th/5-3/9.16-1-math-zimu-shu1";
//	string videodir = "/home/data/jiangbo/xiaoxue/arranged/qian/1th/6-2/16";
//	string resultdir = "/home/data/Class_results/arranged/qian/1th/6-2/16";
//	if (!fs::IsExists(resultdir)){
//		fs::MakeDir(resultdir);
//	}
//	vector<string>videolist = fs::ListDir(videodir, { "mp4" });
//	for (int i = 0; i < videolist.size(); i++){
//		//string videoname = videolist[i];
//		initValue(n, max_student_num, class_info_all, student_valid, students_all);
//		string videoname = "math.mp4";
//		//string output = "../output/" + videoname;
//		string output = resultdir + "/" + videoname;
//		if (!fs::IsExists(output)){
//			fs::MakeDir(output);
//		}
//		string videopath = videodir+"/" + videoname;
//		VideoCapture capture(videopath);
//		if (!capture.isOpened())
//		{
//			printf("video loading fail");
//		}
//		long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
//		cout << "all " << totalFrameNumber << " frame" << endl;
//		//long frameToStart = 13500; 
//		//capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
//
//		Mat frame;
//		int n = 0;
//		std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
//		while (true)
//		{
//			if (!capture.read(frame)){
//				break;
//			}
//			
//			if (n < 25 * 10){
//				cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
//				PoseInfo pose1;
//				GetStandaredFeats(net1, pose1, frame, n, output,max_student_num);
//			}
//
//			else{
//				cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
//				PoseInfo pose;
//				student_info = student_detect(net1, frame, n, pose, output);
//				/*vector<vector<Student_Info>>students_all = get<0>(student_info);
//				vector<Class_Info>class_info_all = get<1>(student_info);*/
//			}
//			n++;
//		}
//	}
//
//	//--------------use pic to test----------------------------
//	
//	/*vector<string>imagelist = fs::ListDir("/home/lw/student_api/inputimg", { "jpeg" });
//	int n = 0;
//	for (int i = 0; i < imagelist.size(); i++){
//		string imagep = "/home/lw/student_api/inputimg/" + imagelist[i];
//		Mat image = imread(imagep);
//		cv::resize(image, image, Size(0, 0), 2 / 3., 2 / 3.);
//		PoseInfo pose;
//		vector<Student_Info>student_info;
//		student_info = student_detect(net1, image, n, pose);
//		n++;
//	}*/
//	
//}

