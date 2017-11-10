#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "student/student.hpp"
#include <numeric>

using namespace cv;
using namespace std;
using namespace caffe;
vector<float> linspace(float a, float b, int c){
	vector<float>line;
	float delta = float(b - a) / (c - 1);
	for (int i = 0; i < c; i++){
		line.push_back(a + i*delta);
	}
	return line;
}
bool greate(vector<float>a, vector<float>b){
	return a[2] > b[2];
}
PoseInfo pose_detect(Net &net,Mat &oriImg,PoseInfo &pose){
	Timer timer;
	float scale = 1.5*368.0 / oriImg.size[0];
	//float scale = 1.0;
	Mat imagetotest;
	cv::resize(oriImg, imagetotest, Size(0, 0), scale, scale);
	vector<Mat> bgr;
	cv::split(imagetotest, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1/256.f, -0.5);
	bgr[1].convertTo(bgr[1], CV_32F, 1/256.f, -0.5);
	bgr[2].convertTo(bgr[2], CV_32F, 1/256.f, -0.5);
	
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, imagetotest.rows, imagetotest.cols);
	
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	net.Forward();

	shared_ptr<Blob> output_blobs = net.blob_by_name("Mconv7_stage6_L2");
	shared_ptr<Blob> output_blobs1 = net.blob_by_name("Mconv7_stage6_L1");
	Mat heatmap = Mat::zeros(output_blobs->height(), output_blobs->width(), CV_32FC(19));
	Mat paf_avg = Mat::zeros(output_blobs1->height(), output_blobs1->width(), CV_32FC(38));
	for (int i = 0; i < output_blobs->channels(); i++){
		for (int j = 0; j < output_blobs->height(); j++){
			for (int k = 0; k < output_blobs->width(); k++){
				heatmap.at<float>(j, 19*k+i) = output_blobs->data_at(0, i, j, k);		
			}
		}
	}
	for (int i = 0; i < output_blobs1->channels(); i++){
		for (int j = 0; j < output_blobs1->height(); j++){
			for (int k = 0; k < output_blobs1->width(); k++){
				paf_avg.at<float>(j, 38 * k + i) = output_blobs1->data_at(0, i, j, k);
			}
		}
	}
	//cv::resize(heatmap, heatmap, Size(0, 0), stride, stride);
	cv::resize(heatmap, heatmap, cv::Size(oriImg.size[1], oriImg.size[0]));
	//cv::resize(paf_avg, paf_avg, Size(0, 0), stride, stride);
	cv::resize(paf_avg, paf_avg, cv::Size(oriImg.size[1], oriImg.size[0]));
	/*for (int i = 0; i < 9; i++){
		cout << heatmap.at<float>(800, i) << "  " << endl;
	}*/
	Mat compare1, compare2, compare3, compare4, compare5;
	Mat bool_1, bool_2, bool_3, bool_4;
	Mat map_1(oriImg.size[0], oriImg.size[1], CV_32F, cv::Scalar::all(0.1));
	vector<float>peaks;
	int peak_counter = 0;
	Point max_loc;
	
	double max_val=0;
	Mat map_ori = Mat::zeros(oriImg.size[0], oriImg.size[1],CV_32F);
	//cout << oriImg.size[0] << oriImg.size[1] << endl;
	

	for (int i = 0; i < 8; i++){
		for (int j = 0; j < oriImg.size[0]; j++){
			for (int k = 0; k < oriImg.size[1]; k++){
				map_ori.at<float>(j, k) = heatmap.at<float>(j, 19 * k + i);
			}
		}
		
		Mat map;
		GaussianBlur(map_ori, map, Size(7,7),3, 3);

		Mat map_left = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.rowRange(0, oriImg.size[0] - 1).copyTo(map_left.rowRange(1, oriImg.size[0]));

		Mat map_right = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.rowRange(1, oriImg.size[0]).copyTo(map_right.rowRange(0, oriImg.size[0] - 1));

		Mat map_up = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.colRange(0, oriImg.size[1] - 1).copyTo(map_up.colRange(1, oriImg.size[1]));

		Mat map_down = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.colRange(1, oriImg.size[1]).copyTo(map_down.colRange(0, oriImg.size[1] - 1));

		//cout << map.rowRange(0, oriImg.size[0] - 1) << endl;
		//cout << map_left.rowRange(1, oriImg.size[0]) << endl;

		compare(map, map_left, compare1, CMP_GE);
		compare(map, map_right, compare2, CMP_GE);
		compare(map, map_up, compare3, CMP_GE);
		compare(map, map_down, compare4, CMP_GE);
		compare(map, 0.1, compare5, CMP_GT);
		
		bitwise_and(compare1, compare2, bool_1);
		bitwise_and(compare3, bool_1, bool_2);
		bitwise_and(compare4, bool_2, bool_3);
		bitwise_and(compare5, bool_3, bool_4);

		for (int j = 0; j < bool_4.rows; j++){
			for (int k = 0; k < bool_4.cols; k++){
				if (int(bool_4.at<uchar>(j, k)) == 255){
					peaks.push_back(k);
					peaks.push_back(j);
					peaks.push_back(map_ori.at<float>(j, k));
					peaks.push_back(peak_counter);
					peak_counter++;
				}
			}
		}
		
		pose.all_peaks.push_back(peaks);
		peaks.clear();

		/*minMaxLoc(bool_4, 0, 0, 0, &max_loc);
		cout << bool_4.at<float>(max_loc.y, max_loc.x) << endl;
		peaks.push_back(max_loc.x);
		peaks.push_back(max_loc.y);
		peaks.push_back(map_ori.at<float>(max_loc.y, max_loc.x));
		pose.all_peaks.push_back(peaks);
		peaks.clear();*/
		
	}
	int limbSeq[19][2] = { { 2, 1 }, { 2, 3 }, { 2, 6 }, { 3, 4 }, { 4, 5 }, { 6, 7 }, { 7, 8 }, { 2, 9 }, { 9, 10 }, { 10, 11 }, { 2, 12 }, { 12, 13 }, { 13, 14 }, { 1, 15 }, { 15, 17 }, { 1, 16 }, { 16, 18 }, { 3, 17 }, { 6, 18 } };
	int mapIdx[19][2] = { { 47, 48 }, { 31, 32 }, { 39, 40 }, { 33, 34 }, { 35, 36 }, { 41, 42 }, { 43, 44 }, { 19, 20 }, { 21, 22 }, { 23, 24 }, { 25, 26 }, { 27, 28 }, { 29, 30 }, { 49, 50 }, { 53, 54 }, { 51, 52 }, { 55, 56 }, { 37, 38 }, { 45, 46 } };
	vector<vector<vector<float>>>connection_all;
	vector<int>special_k;
	int mid_num = 10;
	vector<float>aaa{ 1.0 };
	vector<vector<float>>bbb;
	bbb.push_back(aaa);
	Mat score_mid1 = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
	Mat score_mid2 = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
	for (int i = 0; i < 7; i++){
	
	//for (int i = 0; i < 2; i++){
		for (int j = 0; j < oriImg.size[0]; j++){
			for (int k = 0; k < oriImg.size[1]; k++){
				score_mid1.at<float>(j, k) = paf_avg.at<float>(j, 38 * k + (mapIdx[i][0] - 19));
				score_mid2.at<float>(j, k) = paf_avg.at<float>(j, 38 * k + (mapIdx[i][1] - 19));
			}
		}
		
		vector<float>candA = pose.all_peaks[limbSeq[i][0] - 1];
		vector<float>candB = pose.all_peaks[limbSeq[i][1] - 1];
		int nA = candA.size()/4;
		int nB = candB.size()/4;
		
		vector<float>vec;
		vector<float>startend1;
		vector<float>startend2;
		
		if (nA != 0 && nB != 0){
			vector<vector<float>>connection_candidate;
			
			vector<float>connection_can;
			for (int j = 0; j < nA; j++){
				for (int k = 0; k < nB; k++){
					vec.clear();
					vec.push_back(candA[4 * j] - candB[4 * k]);
					vec.push_back(candA[4 * j + 1] - candB[4 * k + 1]);
					float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
					norm = norm + 1e-5;
					vec[0] = float(vec[0]) / norm;
					vec[1] = float(vec[1]) / norm;
					startend1 = linspace(candA[4 * j], candB[4 * k], mid_num);
					startend2 = linspace(candA[4 * j + 1], candB[4 * k + 1], mid_num);
					
					vector<float>vec_x;
					vector<float>vec_y;
					for (int l = 0; l < mid_num; l++){
						vec_x.push_back(score_mid1.at<float>(int(round(startend2[l])), int(round(startend1[l]))));
						vec_y.push_back(score_mid2.at<float>(int(round(startend2[l])), int(round(startend1[l]))));
					}
					vector<float>score_midpts;
					for (int l = 0; l < mid_num; l++){
						vec_x[l] = vec_x[l] * vec[0];
						vec_y[l] = vec_y[l] * vec[1];
						score_midpts.push_back(-(vec_x[l] + vec_y[l]));
					}
					/*float sum = 0.0;
					for (int k = 0; k < score_midpts.size(); k++){
						sum += score_midpts[k];
					}
					cout << sum << " ";*/
					float score_with_dist_prior = accumulate(score_midpts.begin(), score_midpts.end(), 0.0) / 10.0 + MIN(0.5*oriImg.size[0] / norm - 1, 0.0);
					/*cout << accumulate(score_midpts.begin(), score_midpts.end(), 0.0) << " ";*/
					int nonzeronum = 0;
					int criterion1 = 0;
					int criterion2 = 0;
					for (int l = 0; l < score_midpts.size(); l++){
						if (score_midpts[l] > 0.05){
							nonzeronum += 1;
						}
					}
					
					if (nonzeronum > 8){
						criterion1 = 1;
					}
					if (score_with_dist_prior > 0.0){
						criterion2 = 1;
					}
					
					if (criterion1 && criterion2){
						connection_can.push_back(j);
						connection_can.push_back(k);
						connection_can.push_back(score_with_dist_prior);
						connection_can.push_back(score_with_dist_prior + candA[4 * j + 2] + candB[4 * k + 2]);
						connection_candidate.push_back(connection_can);
						connection_can.clear();
					}
				/*	for (int l = 0; l < mid_num; l++){
						cout << vec_x[l] << "  ";
						if (l == 9)cout << endl;
					}*/	
				}
			}
			
			sort(connection_candidate.begin(), connection_candidate.end(), greate);
			
			vector<vector<float>>connection;
			
			vector<float>saveA;
			vector<float>saveB;
			for (int c = 0; c < connection_candidate.size(); c++){
				vector<float>connect;
				float m = connection_candidate[c][0];
				float n = connection_candidate[c][1];
				float s = connection_candidate[c][2];
				auto iter1 = find(saveA.begin(), saveA.end(), m);
				auto iter2 = find(saveB.begin(), saveB.end(), n);
				if (iter1 == saveA.end() && iter2==saveB.end()){
					connect.push_back(candA[4 * m + 3]);
					connect.push_back(candB[4 * n + 3]);
					connect.push_back(s);
					connect.push_back(m);
					connect.push_back(n);
					connection.push_back(connect);
					saveA.push_back(m);
					saveB.push_back(n);
					if (connection.size() >= MIN(nA, nB)){
						break;
					}
				}	
			}
			
			/*for (int j = 0; j < connection.size(); j++){
				for (int k = 0; k < connection[j].size(); k++){
					cout << connection[j][k] << " ";
					if (k == connection[j].size() - 1){
						cout << endl;
					}
				}
			}*/
			connection_all.push_back(connection);
		}
		else{
			special_k.push_back(i);
			connection_all.push_back(bbb);
		}
	}
	
	vector<float>candicate;
	for (int i = 0; i < pose.all_peaks.size(); i++){
		for (int j = 0; j < pose.all_peaks[i].size() / 4; j++){
			candicate.push_back(pose.all_peaks[i][4 * j]);
			candicate.push_back(pose.all_peaks[i][4 * j + 1]);
			candicate.push_back(pose.all_peaks[i][4 * j + 2]);
			candicate.push_back(pose.all_peaks[i][4 * j + 3]);
			pose.candicate.push_back(candicate);
			candicate.clear();
		}
	}
	
	for (int i = 0; i < 7; i++){
	//for (int i = 0; i < 2; i++){
		auto iter = find(special_k.begin(), special_k.end(), i);
		if (iter == special_k.end()){
			vector<float>partAs, partBs;
			for (int j = 0; j < connection_all[i].size(); j++){
				partAs.push_back(connection_all[i][j][0]);
				partBs.push_back(connection_all[i][j][1]);
			}
			int indexA = limbSeq[i][0] - 1;
			int indexB = limbSeq[i][1] - 1;
			
			for (int j = 0; j < connection_all[i].size(); j++){
				int found = 0;
				int subset_idx[2] = { -1, -1 };
				for (int k = 0; k < pose.subset.size(); k++){
					if (pose.subset[k][indexA] == partAs[j] || pose.subset[k][indexB] == partBs[j]){
						subset_idx[found] = k;
						found += 1;
					}
				}
				
				if (found == 1){
					int idx = subset_idx[0];
					if (pose.subset[idx][indexB] != partBs[j]){
						pose.subset[idx][indexB] = partBs[j];
						pose.subset[idx][18] += pose.candicate[partBs[j]][2] + connection_all[i][j][2];
						pose.subset[idx][19] += 1;
					}
				}
				if (found == 0){
					vector<float>sub(20, -1);
					sub[indexA] = partAs[j];
					sub[indexB] = partBs[j];
					sub[18] = pose.candicate[connection_all[i][j][0]][2] + pose.candicate[connection_all[i][j][1]][2] + connection_all[i][j][2];
					sub[19] = 2;
					pose.subset.push_back(sub);
				}
			}
		}
	}
	return pose;
}