#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "ImgFeature.h"
#include "SurfFeature.h"
#include "Utils.h"
#include "NearestNeighbors.h"
#include "ImgConAna.h"
#include "ostuRemoveBorder.h"

#include <time.h>

using namespace std;

const int surfDescriptorSize = 64;
const int denseSurfFeatureSize = 624 * surfDescriptorSize;
const int clusterCount = 20;
const int surfVladFeatureSize = clusterCount * surfDescriptorSize; 


void learn() {
	//图片库
	string filepath = "D:\\SearchPictures\\2\\train";
	vector<string> files;
	getFiles(filepath, files);
	cout << "Files read end." << endl;

	clock_t start;
	start = clock();
	
	int featureMatRows = 0;
	vector<float* > featureArray;
	size_t imgNums = files.size();
	for (size_t n = 0; n < imgNums; ++n) {
		Mat imgSrc = imread(files[n]);

		vector<Mat> proposals;
		proposalMat(imgSrc,proposals);
		size_t pNums = proposals.size();
		featureMatRows += pNums;
		for (size_t i = 0; i < pNums; ++i) {
			IplImage img = IplImage(proposals[i]);
			// 归一化图片
			IplImage* stdImg= cvCreateImage( cvSize(240,260), IPL_DEPTH_8U, 3);
			cvResize(&img, stdImg, CV_INTER_LINEAR);
			// 提取Surf特征
			SurfFeature sfeature;
			sfeature.computeFeature(stdImg);
			float* feature = sfeature.GetFeature();
			featureArray.push_back(feature);
			cout << files[n] << ":" << i << " end." << endl;
			cvReleaseImage(&stdImg);
		}
	}
	Mat featureMat(featureMatRows, surfDescriptorSize, CV_32F);
	for (int r = 0; r < featureMatRows; ++r) {
		float* feature = featureArray[r];
		for (int c = 0; c < surfDescriptorSize; ++c) {
			featureMat.at<float>(r, c) = *feature;
			++feature;
		}
	}
	// 释放Surf特征
	for (int i = 0 ; i < featureMatRows; ++i) freeArray(featureArray[i]);
	
	// K-means聚类码本
	Mat centers(clusterCount, 1, featureMat.type());
	Mat labels;
	double cretia = kmeans(featureMat, clusterCount, labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 50, 0.1), 3, KMEANS_PP_CENTERS, centers);
	cout << "紧凑度： " << cretia << endl;
	//写码本
	ofstream outfile("D:\\SearchPictures\\2\\features\\centers.txt");
	outfile.precision(8);
	for (int r = 0; r < clusterCount; ++r) {
		for (int c = 0; c < surfDescriptorSize; ++c) {
			outfile << centers.at<float>(r, c) << " ";
		}
		outfile << endl;
	}
	outfile.close();
	//string binaryOutfile = "..\\..\\center.bin";
	//mat2Bin(binaryOutfile.c_str(), featureMat);
	clock_t end = clock();
	cout << "Total Time: " << (float)(end - start) / CLOCKS_PER_SEC << endl;
}

void train() {
	//Load 码本
	string vocabularyFile = "D:\\SearchPictures\\2\\features\\centers.txt";
	float** vocabulary = newMatrix(clusterCount, surfDescriptorSize);
	loadVocabulary(vocabularyFile, vocabulary, clusterCount, surfDescriptorSize);

	string filepath = "D:\\SearchPictures\\2\\train";
	vector<string> files;
	getFiles(filepath, files);

	//图片库路径表
	ofstream imagePathOutfile("D:\\SearchPictures\\2\\features\\imagePath.txt");
	//ofstream outfile("..\\..\\surfValdFeature.txt");
	//图片库特征表
	FILE* fin;
	fin = fopen("D:\\SearchPictures\\2\\features\\surfValdFeature.bin", "wb");

	size_t imgNums = files.size();
	imagePathOutfile << imgNums << endl;
	for (size_t n = 0; n < imgNums; ++n) {
		Mat imgSrc = imread(files[n]);

		vector<Mat> proposals;
		proposalMat(imgSrc, proposals);
		size_t pNums = proposals.size();
		imagePathOutfile << files[n] << endl;
		imagePathOutfile << pNums << endl;

		for (size_t i = 0; i < pNums; ++i) {
			Mat stdImg;
			resize(proposals[i], stdImg, Size(240, 260));
			// 提取Surf&VLAD特征
			ImgFeature feature(stdImg);
			feature.computeSurfVladFeature(vocabulary, clusterCount, surfDescriptorSize);
			float* featureArray = feature.GetFeature();
			feature2Bin(featureArray, clusterCount * surfDescriptorSize, fin);
			//for (int j = 0; j < clusterCount * surfDescriptorSize; ++j) outfile << featureArray[j] << " ";
			//outfile << endl;
			cout << files[n] << ":" << i << " end." << endl;
			feature.FreeFeature(featureArray);
		}
	}
	imagePathOutfile.close();
	fclose(fin);
}

void test() {
	// load vocabulary
	string vocabularyFile = "E:\\AnXingle\\Search_Project\\QiChengZuo\\fileFeatures\\centers.txt";
	//加载码本
	float** vocabulary = newMatrix(clusterCount, surfDescriptorSize);
	loadVocabulary(vocabularyFile, vocabulary, clusterCount, surfDescriptorSize);
	cout << "Load Vocabulary End. " << endl;

	// Load train image feature
	string trainFile = "E:\\AnXingle\\Search_Project\\QiChengZuo\\fileFeatures\\imagePath.txt";
	string featureFile = "E:\\AnXingle\\Search_Project\\QiChengZuo\\fileFeatures\\surfValdFeature.bin";
	unordered_map<string, vector<float*> > trainFeatureMap;
	loadFeatureBin(trainFile, featureFile, surfVladFeatureSize, trainFeatureMap);
	cout << "Load Train Feature End: " << trainFeatureMap.size() << endl;

	// Load test file 
	/*************************          需要知道 testFiles的结构           *************************/
	/********************          testFiles  E:\\AnXingle\\Search_Project\\z1.jpg        *********/
	string testPath = "E:\\AnXingle\\Search_Project\\QiChengZuo\\test";
	vector<string> testFiles;
	getFiles(testPath, testFiles);

	int K = 6;
	string rootPath = "..\\..\\QiChengZuo_result\\";
	system("rd /s/q ..\\..\\QiChengZuo_result");
	system("md ..\\..\\QiChengZuo_result");

	// extract test image feature && find K-nearest images
	size_t testNum = testFiles.size();
	for (size_t t = 0; t < testNum; ++t)
	{
		Mat imgSrc = imread(testFiles[t]);
		vector<Mat> proposals;
		proposalMat(imgSrc, proposals);
		size_t pNums = proposals.size();
		// extract features
		vector<float* > testFeatures;
		for (size_t i = 0; i < pNums; ++i) {
			Mat stdImg;
			resize(proposals[i], stdImg, Size(240, 260));
			// 提取Surf&VLAD特征
			ImgFeature feature(stdImg);
			feature.computeSurfVladFeature(vocabulary, clusterCount, surfDescriptorSize);
			float* featureArray = feature.GetFeature();
			testFeatures.push_back(featureArray);
		}
		/*************      写入result文件夹，test中的测试文件         *************/
		string outPath = rootPath + intToString(t + 1);
		string mkidr = "md " + outPath;
		system(mkidr.c_str());
		string newTestPath = outPath + "\\" + splitFileName(testFiles[t]);
		imwrite(newTestPath.c_str(), imgSrc);

		// K-nearest neighbors search
		/*******************       计算最近邻的匹配结果图片             **********************/
		vector<pair<string, double> > disMap;
		computeKNearestImage(testFeatures, trainFeatureMap, surfVladFeatureSize, K, disMap);
		cout << disMap.size() << endl;

		//string disFilePath = outPath + "\\" + "dis.txt";
		//ofstream disFile(disFilePath.c_str());
		for (int i = 0; i < K; ++i)
		{
			//disFile << (i + 1) << " : " << disMap[i].second << endl;
			/**********     trainImagePath   E:\\Anxingle\\Search_Project\\train\\123.jpg     **************/
			/*************          trainImageName   123.jpg                    ****************************/
			string trainImagePath = disMap[i].first;
			string trainImageName = splitFileName(trainImagePath);
			IplImage *img = cvLoadImage((trainImagePath.c_str()), CV_LOAD_IMAGE_COLOR);
			string newImagePath = outPath + "\\" + intToString(i + 1) + "_" + trainImageName;
			cvSaveImage(newImagePath.c_str(), img);
			cvReleaseImage(&img);
		}
		//disFile.close();
		// release
		freeVectorArray(testFeatures);
		cout << "Test " << t << " End." << endl;
	}
	// release
	freeMatrix(vocabulary, clusterCount);
	freeFeatureMap(trainFeatureMap);
}

int main()
{
	learn();
	train();
	//test();
	system("pause");
	return 0;
}