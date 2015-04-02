#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>  
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <direct.h>
#include <fstream>
#include "ml.h"
#include <utility>
#include <list>
#include <io.h>
#include <time.h>
#include <algorithm>

#include "SBE.h"

#define ELLIPSOID_SIGMA		6 //4 //4.5
#define ELLIPSOID_A			1
#define ELLIPSOID_B			1
#define ELLIPSOID_C			1.25
#define EXCHANGE_FREQUENCY	5
#define STUDY_NUM			20
#define rndSize				49999

#define CHECKPOINT			1/2 //3/4 // 2/3
#define CHECKVARIANCE		3

#define MAXAXIS				16

using namespace std;
using namespace cv;

int rndn = 0;
int rnd3[rndSize];
int rnd10[rndSize];
int getrndn(){
	rndn++;
	if (rndn >= rndSize)
		rndn = 0;
	return rndn;
}

SBE::SBE(int img_rows, int img_cols)
{
	RNG rnd = theRNG();
	for (int i = 0; i < rndSize; i++){
		rnd3[i] = rnd(3) - 1;
		rnd10[i] = rnd(10);
	}

	imgNums = 0;
	imgRols = img_rows;
	imgCols = img_cols;
	for (int i = 0; i < imgRols * imgCols; i++)
	{
		codeData.push_back(vector<Vec3d>());
		codeEllipsoid.push_back(vector<StructEllipsoid>());
	}

	// initial
	globalMinR = -1;
}

SBE::~SBE()
{

}

// --------- Public --------------------------------------

Mat SBE::Run(Mat &image){

	Mat image_result = Mat::zeros(imgRols, imgCols, CV_8UC1);
	imgNums++;

	for (int i = 0; i < imgRols; i++) {
		for (int j = 0; j < imgCols; j++) {
			
			// for each pixel
			int pixel = i * imgCols + j;
			if (codeEllipsoid[pixel].size() == 0){
				// INITIAL THE PIXEL
				// 1. prepare data
				if (codeData[pixel].size() == 0){
					// create initial data
					vector<Vec2f> npointoffset;
					npointoffset.push_back(Vec2f(0,0));			// itself
					GetNeighborPosOffset(1, 8, npointoffset);	// ceil to one around
					for (int n = 0; n < npointoffset.size(); n++){
						int newi = i + (int)(npointoffset[n][0]);
						int newj = j + (int)(npointoffset[n][1]);
						if (newi >= 0 && newi < imgRols && newj >= 0 && newj < imgCols){
							Vec3d temp = (Vec3d)image.at<Vec3b>(newi, newj);
							if (!DetectCombine(codeData[pixel], temp)){
								codeData[pixel].push_back(temp);		// 0: itself; 1-8: one round; 9-24: two round
							}
						}
					}
				}
				else{
					// add new data
					Vec3d temp = (Vec3d)image.at<Vec3b>(i, j);
					if (!DetectCombine(codeData[pixel], temp))
						codeData[pixel].push_back(temp);				// only add itself
				}

				// 2. calculate ellipsoid
				CreateEllipsoid(codeData[pixel], codeEllipsoid[pixel]);
			}
			else{
				// DETECT THE PIXEL
				Vec3d temp = (Vec3d)image.at<Vec3b>(i, j);
				int k = 0;
				float dist = DetectPixel(k, codeEllipsoid[pixel], temp);
				if (dist >= 0){
					// have ellipsoids here
					if (dist < 1){
						image_result.at<unsigned char>(i, j) = 0;
						// in a ellipsoid and update it's mean
						UpdateEllipsoid(temp, codeEllipsoid[pixel][k]);		// update myself
					}
					else{
						image_result.at<unsigned char>(i, j) = 255;
						// out all ellipsoid and transmit all!
						vector<Vec2f> npointoffset;
						npointoffset.push_back(Vec2f(0, 0));			// itself
						GetNeighborPosOffset(1, 8, npointoffset);		
						GetNeighborPosOffset(2, 16, npointoffset);
						for (int n = 0; n < npointoffset.size(); n++){	// for each neighbor
							int newi = i + (int)(npointoffset[n][0]);
							int newj = j + (int)(npointoffset[n][1]);
							if (newi >= 0 && newi < imgRols && newj >= 0 && newj < imgCols){
								int pixel_nbr = newi * imgCols + newj;
								int nsize = codeEllipsoid[pixel_nbr].size();
								if (nsize > 0){
									int el_nbr = getRandomNum(nsize);
									StructEllipsoid tempel = codeEllipsoid[pixel_nbr][el_nbr];
									AddEllipsoid(codeEllipsoid[pixel], tempel);
								}
							}
						}
					}

					// check useless ellipsoid and delete it
					DeleteEllipsoid(codeEllipsoid[pixel], imgNums);
				}
				else{
					// no ellipsoid here
					image_result.at<unsigned char>(i, j) = 0;
				}

			}

		}
	}
	return image_result;

}

// --------- Private --------------------------------------

bool SBE::AddEllipsoid(vector<StructEllipsoid> &sel, StructEllipsoid &el){

	int ss = sel.size();
	for (int i = 0; i < ss; i++){
		float dist = abs(sel[i].mu[0] - el.mu[0]) + abs(sel[i].mu[1] - el.mu[1]) + abs(sel[i].mu[2] - el.mu[2]);
		if (dist < 1){
			if (sel[i].a < el.a) sel[i].a = el.a;
			if (sel[i].b < el.b) sel[i].b = el.b;
			if (sel[i].c < el.c) sel[i].c = el.c;
			return true;
		}
	}
	sel.push_back(el);
	return true;
}

bool SBE::UpdateEllipsoid(Vec3d data_now, StructEllipsoid &el){

	// only update means
	el.mu[0] = (el.mu[0] * el.n + data_now[0]) / (el.n + 1);
	el.mu[1] = (el.mu[1] * el.n + data_now[1]) / (el.n + 1);
	el.mu[2] = (el.mu[2] * el.n + data_now[2]) / (el.n + 1);

	el.n++;

	return true;
}

int SBE::getRandomNum(int ssize){

	if (ssize == 0){
		return rnd10[getrndn()];;
	}
	else if (ssize / 10 > 0){
		int ti = ssize / 10;
		int reti = getRandomNum(ti);
		int gi = ssize % 10;
		int regi = getRandomNum(gi);
		return reti * 10 + regi;
	}
	else{
		int re;
		do {
			re = rnd10[getrndn()];
		} while ( re >= ssize );
		return re;
	}

}

int SBE::GetNeighborPos(int i, int j, int rows, int cols){

	Vec2i re;

	re[0] = rnd3[getrndn()] + i;
	re[1] = rnd3[getrndn()] + j;
	while ((re[0] * cols + re[1]) < 0 || (re[0] * cols + re[1]) > rows * cols - 1 || (re[0] == 0 && re[1] == 0)){
		re[0] = rnd3[getrndn()] + i;
		re[1] = rnd3[getrndn()] + j;
	}

	return re[0] * cols + re[1];

}

void SBE::GetNeighborPosOffset(int r, int n, vector<Vec2f> &npointoffset){

	float perangle = 6.2832 / n;
	// for each point
	for (int i = 0; i < n; i++){
		float px = r * sin(perangle * i);
		float py = r * cos(perangle * i);
		npointoffset.push_back(Vec2f(px, py));
	}

}

float SBE::GetMinAxis(StructEllipsoid &el){

	float remin = el.a;
	if (el.b < remin)
		remin = el.b;
	if (el.c < remin)
		remin = el.c;

	if (globalMinR > remin || globalMinR < 0)
		globalMinR = remin;

	return remin;

}

bool SBE::DetectCombine(vector<Vec3d> &vec_pixel, Vec3d now_pixel){

	int ss = vec_pixel.size();
	for (int i = 0; i < ss; i++){
		if (now_pixel[0] == vec_pixel[i][0] && now_pixel[1] == vec_pixel[i][1] && now_pixel[2] == vec_pixel[i][2]){
			return true;
		}
	}
	return false;
}

bool SBE::CheckPixelData_Medium(vector<Vec3d> &input_data, vector<Vec3d> &output_data){

	// calculate 
	const int ss = input_data.size();
	double *array_x = new double[ss];
	double *array_y = new double[ss];
	double *array_z = new double[ss];
	for (int i = 0; i < ss; i++){
		array_x[i] = input_data[i][0];
		array_y[i] = input_data[i][1];
		array_z[i] = input_data[i][2];
	}

	sort(array_x, array_x + ss);
	sort(array_y, array_y + ss);
	sort(array_z, array_z + ss);

	int tp = floor((ss - 1) / 2);
	double med[3] = { array_x[tp], array_y[tp], array_z[tp] };

	for (int i = 0; i < ss; i++){
		array_x[i] = abs(array_x[i] - med[0]);
		array_y[i] = abs(array_y[i] - med[1]);
		array_z[i] = abs(array_z[i] - med[2]);
	}

	sort(array_x, array_x + ss);
	sort(array_y, array_y + ss);
	sort(array_z, array_z + ss);

	double K = 1.4826;
	double mad[3] = { array_x[tp], array_y[tp], array_z[tp] };
	double sigma[3] = { max(mad[0] * K, K * CHECKVARIANCE), max(mad[1] * K, K * CHECKVARIANCE), max(mad[2] * K, K * CHECKVARIANCE) };

	// detect
	for (int i = 0; i < ss; i++){
		if ( (input_data[i][0] > (med[0] - sigma[0])) && (input_data[i][0] < (med[0] + sigma[0])) ){
			if ((input_data[i][1] > (med[1] - sigma[1])) && (input_data[i][1] < (med[1] + sigma[1]))){
				if ((input_data[i][2] > (med[2] - sigma[2])) && (input_data[i][2] < (med[2] + sigma[2]))){
					output_data.push_back(input_data[i]);
				}
			}
		}
	}
	if (output_data.size() > 0)
		return true;
	else
		return false;
}

bool SBE::CreateEllipsoid(vector<Vec3d> &code_data, vector<StructEllipsoid> &code_ellipsoid){

	// Make Sure Only Use The Data
	if (code_data.size() < 4)
		return false;


	vector<Vec3d> datas;
	if (code_data.size() > 20){
		const int ss = code_data.size();
		double *array_x = new double[ss];
		double *array_y = new double[ss];
		double *array_z = new double[ss];
		for (int i = 0; i < ss; i++){
			array_x[i] = code_data[i][0];
			array_y[i] = code_data[i][1];
			array_z[i] = code_data[i][2];
		}

		sort(array_x, array_x + ss);
		sort(array_y, array_y + ss);
		sort(array_z, array_z + ss);

		int tp = floor((ss - 1) / 2);
		double med[3] = { array_x[tp], array_y[tp], array_z[tp] };
		datas.push_back(Vec3d(med[0], med[1], med[2]));
	}
	else{
		// check the data cluster enough or not
		if (!CheckPixelData_Medium(code_data, datas))
			return false;
	}

	StructEllipsoid el;
	int ss = datas.size();
	if (ss > 3){

		// calculate ellipsoid
		el.n = 0;
		float sigmax[3], sigmax2[3]; 
		sigmax[0] = sigmax[1] = sigmax[2] = 0;
		sigmax2[0] = sigmax2[1] = sigmax2[2] = 0;
		for (int j = 0; j < ss; j++){
			el.n++;
			sigmax[0] += datas[j][0];
			sigmax[1] += datas[j][1];
			sigmax[2] += datas[j][2];
			sigmax2[0] += datas[j][0] * datas[j][0];
			sigmax2[1] += datas[j][1] * datas[j][1];
			sigmax2[2] += datas[j][2] * datas[j][2];
		}
		if (CalcEllipsoid(el, sigmax, sigmax2)){
			// update minR
			if (globalMinR < 0)
				globalMinR = GetMinAxis(el);
			else{
				float minnow = GetMinAxis(el);
				if (globalMinR > minnow)
					globalMinR = minnow;
			}
			code_ellipsoid.push_back(el);
			return true;
		}
	}
	else if (imgNums > 20){
		CreateSphere(el, datas);
		code_ellipsoid.push_back(el);
		return true;
	}
	return false;

}

bool SBE::CalcEllipsoid(StructEllipsoid &el, float* sigmax, float* sigmax2){

	// re-calculate
	el.mu[0] = sigmax[0] / el.n;
	el.mu[1] = sigmax[1] / el.n;
	el.mu[2] = sigmax[2] / el.n;

	float cov[3];
	cov[0] = el.mu[0] * el.mu[0] - sigmax2[0] / el.n;
	cov[1] = el.mu[1] * el.mu[1] - sigmax2[1] / el.n;
	cov[2] = el.mu[2] * el.mu[2] - sigmax2[2] / el.n;

	el.a = sqrt(abs(cov[0])) * ELLIPSOID_SIGMA * ELLIPSOID_A;
	el.b = sqrt(abs(cov[1])) * ELLIPSOID_SIGMA * ELLIPSOID_B;
	el.c = sqrt(abs(cov[2])) * ELLIPSOID_SIGMA * ELLIPSOID_C;

	if (el.a == 0 || el.b == 0 || el.c == 0)
		return false;
	if (el.a > MAXAXIS || el.b > MAXAXIS || el.c > MAXAXIS)
		return false;

	return true;
}

float SBE::DetectPixel(int &k, vector<StructEllipsoid> &code_ellipsoid, Vec3d code_data){

	// [0 - n]	means here return the small dist number between the code_data and an ellipsoid.
	// -1		means no ellipsoid here
	float dist = -1;
	int ss = code_ellipsoid.size();
	for (int i = 0; i < ss; i++){
		float temp = pow((code_data[0] - code_ellipsoid[i].mu[0]) / code_ellipsoid[i].a, 2) +
			pow((code_data[1] - code_ellipsoid[i].mu[1]) / code_ellipsoid[i].b, 2) +
			pow((code_data[2] - code_ellipsoid[i].mu[2]) / code_ellipsoid[i].c, 2);
		if (dist > temp || dist < 0)
			k = i;
			dist = temp;
	}
	return dist;
}

bool SBE::CreateSphere(StructEllipsoid &sel, vector<Vec3d> sdata){

	float tt[3];
	tt[0] = tt[1] = tt[2] = 0;
	int ss = sdata.size();
	for (int i = 0; i < ss; i++){
		tt[0] += sdata[i][0];
		tt[1] += sdata[i][1];
		tt[2] += sdata[i][2];
	}

	sel.n = 4;
	sel.mu[0] = tt[0] / ss;
	sel.mu[1] = tt[1] / ss;
	sel.mu[2] = tt[2] / ss;
	sel.a = sel.b = sel.c = globalMinR * 6;

	return true;
}

bool SBE::DeleteEllipsoid(vector<StructEllipsoid> &code_ellipsoid, int bg_nums){

	int ss = code_ellipsoid.size();
	for (int i = 0; i < ss; i++){
		float rate = (float)(code_ellipsoid)[i].n / (float)bg_nums;
		if (rate < 0.02 && bg_nums > 100){
			code_ellipsoid.erase(code_ellipsoid.begin() + i);
			ss--;
			i--;
		}
	}
	return true;

}

