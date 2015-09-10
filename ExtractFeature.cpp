#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "SurfFeature.h"
#include "Utils.h"
#include "NearestNeighbors.h"
using namespace std;

#define K_MEANS 6

const int featureSize = 624 * 64;



void train() {
	//图片库
	// outfile  train()
	ofstream outfile("C:\\Users\\CCK\\Desktop\\debug\\mfc_SurfFeature.txt");//**************************************************
	string filepath = "E:\\AnXingle\\WorkSpace\\Test_Picture\\remove_border\\mfc";//**************************************************
	vector<string> files;
	getFiles(filepath, files);
	//各个图片库的特征描述 TXT
	
	outfile.precision(10);

	size_t n = files.size();
	cout << "All have: " << endl;
	cout << n << endl;
	for (size_t i = 0; i < n; ++i) 
	{
		outfile << files[i] << endl;
		//读入图片
		IplImage *img = cvLoadImage((files[i].c_str()), CV_LOAD_IMAGE_COLOR);

		//提取surf特征
		SurfFeature sfeature;
		
		sfeature.computeFeature(img);
		
		float* feature = sfeature.GetFeature();
		
		for(int index = 0; index < featureSize; index++)
		{
			outfile<<feature[index]<<" ";
		}
		outfile<<endl;
		cout << i << endl;
		cout << files[i] << " End." << endl;
		sfeature.FreeFeature(feature);

		cvReleaseImage(&img);
	}
	outfile.close();
}

void test() {
	// 各个图片库的特征描述 TXT
	//featureFile Test()
	string featureFile = "C:\\Users\\CCK\\Desktop\\debug\\mfc_SurfFeature.txt";//**************************************************
	unordered_map<string, string> trainImageMap;
	unordered_map<string, vector<pair<string, float*> > > trainFeatureMap;
	LoadFeature(featureFile, featureSize, trainFeatureMap, trainImageMap);
	cout << "Load train feature end: " << trainFeatureMap.size() << endl;

	// 测试图片文件
	string testPath = "E:\\AnXingle\\WorkSpace\\Test_Picture\\remove_border\\Test_mfc";//*************************************************************************
	// testFiles：该文件夹下的所有图片们
	vector<string> testFiles;
	getFiles(testPath, testFiles);

	// extract test image feature
	unordered_map<string, string> testImageMap;
	unordered_map<string, vector<pair<string, float*> > > testFeatureMap;
	size_t n = testFiles.size();
	for (size_t i = 0; i < n; ++i) {
		string imagePath = testFiles[i];
		cout << "imagePath: ";
		cout << imagePath << endl;
		string imageName = splitFileName(imagePath);
		if (testImageMap.count(imageName) == 0) {
			string prefixPath = imagePath.substr(0, imagePath.find_last_of("_"));
			string suffixPath = imagePath.substr(imagePath.find_last_of("."));
			testImageMap[imageName] = prefixPath + "_0" + suffixPath;
		}

		//读入图片
		IplImage *img = cvLoadImage((imagePath.c_str()), CV_LOAD_IMAGE_COLOR);

		//提取DenseSurf
		SurfFeature surfFeature;
		surfFeature.computeFeature(img);
		float* feature = surfFeature.GetFeature();

		testFeatureMap[imageName].push_back(make_pair(imagePath, feature));
		cout << imagePath << " End." << endl;
		//surfFeature.FreeFeature(feature);
		cvReleaseImage(&img);
	}

	cout << "test map size: " << testImageMap.size() << endl;

	// K近邻 的 K值
	int K = K_MEANS;
	string rootPath = "..\\..\\result_mfc\\";//**********************************************************************************************************
	string rm = "rd " + rootPath;
	string mk = "md" + rootPath;
	//system(rm.c_str());
	//system(mk.c_str());
	// K-nearest image
	int index = 0;
	for (auto& imageMap : testImageMap) {
		++index;
		string outPath = rootPath + intToString(index);
		string mkidr = "md " + outPath;
		system(mkidr.c_str());

		string testImageName = imageMap.first;
		string testImagePath = imageMap.second;
		IplImage *img = cvLoadImage((testImagePath.c_str()), CV_LOAD_IMAGE_COLOR);
		string suffixPath = testImagePath.substr(testImagePath.find_last_of("."));
		string newImagePath = outPath + "\\" + testImageName + suffixPath;
		cvSaveImage(newImagePath.c_str(), img);
		cvReleaseImage(&img);

		vector<pair<string, double> > disMap;
		computeKNearestImage(testFeatureMap[testImageName], featureSize, trainFeatureMap, trainImageMap, K, disMap);//
		cout << "disMap is: " << endl;//  
		cout << disMap.size() << endl;
		for (int i = 0; i < K; ++i) {
			string trainImagePath = disMap[i].first;
			string trainImageName = trainImagePath.substr(trainImagePath.find_last_of("/\\") + 1);
			IplImage *img = cvLoadImage((trainImagePath.c_str()), CV_LOAD_IMAGE_COLOR);
			string newImagePath = outPath + "\\" + intToString(i + 1) + "_"+  trainImageName;
			cvSaveImage(newImagePath.c_str(), img);
			cvReleaseImage(&img);
		}
	}
	freeFeatureMap(trainFeatureMap);
	freeFeatureMap(testFeatureMap);
}

int main()
{
	// train  已经进行到 5  了
	train();
	test();
	//system("pause");

	return 0;
}