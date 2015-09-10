#include <fstream>
#include <cassert> 
#include "SurfFeature.h"
#include "iostream"
using namespace std;

void SurfFeature::computeFeature(IplImage *image_color)
{

	//转换为灰度图像
	IplImage *GrayImage = cvCreateImage(cvGetSize(image_color),IPL_DEPTH_8U,1);
	cvCvtColor(image_color,GrayImage,CV_RGB2GRAY);
	cv::Mat image(GrayImage);

	//提取特征并转换为一维向量
	cv::Mat surfFeature = GetDenseSurf2(image);

	surfFeature.reshape(0,1);
	assert(surfFeatureSize == surfFeature.total());
	featureArray = new float[surfFeatureSize];
	memcpy(featureArray, surfFeature.ptr<float>(0), surfFeatureSize*sizeof(float));

	//释放灰度图像
	cvReleaseImage(&GrayImage);
}

cv::Mat  SurfFeature::GetDenseSurf2(const cv::Mat& image)
{
	int imageRows = image.rows;
	int imageCols = image.cols;

	cv::Mat SurfMat;
	std::vector<cv::KeyPoint> denseKeyPoints;
	if( imageRows >= cwlDiameter && imageCols >= cwlDiameter )
	{
		int row1 = cwlDiameter/2;
		int row2 = imageRows - (cwlDiameter/2) + 1;
		int col1 = cwlDiameter/2;
		int col2 = imageCols - (cwlDiameter/2) + 1;

		for( int row = row1; row <= row2; row += cwlStep )
			for( int col = col1; col <= col2; col += cwlStep )
				denseKeyPoints.push_back(cv::KeyPoint(col,row,cwlDiameter,0));
	
		cv::SurfDescriptorExtractor surfDescriptorExtractor;
		surfDescriptorExtractor.compute(image,denseKeyPoints,SurfMat);

	}
	return SurfMat;
}

cv::Mat  SurfFeature::GetDenseSurf(const cv::Mat& image)
{
	int imageRows = image.rows;
	int imageCols = image.cols;

	cv::Mat SurfMat;
	std::vector<cv::KeyPoint> denseKeyPoints;
	if( imageRows >= cwlDiameter && imageCols >= cwlDiameter )
	{
		int row1 = cwlStep;
		int row2 = imageRows - cwlStep + 1;
		int col1 = cwlStep;
		int col2 = imageCols - cwlStep + 1;

		for( int row = row1; row <= row2; row += cwlStep )
			for( int col = col1; col <= col2; col += cwlStep )
				denseKeyPoints.push_back(cv::KeyPoint(col,row,cwlDiameter,0));

		cv::SurfDescriptorExtractor surfDescriptorExtractor;
		surfDescriptorExtractor.compute(image,denseKeyPoints,SurfMat);
	}
	return SurfMat;
}

float* SurfFeature::GetFeature()
{
	return featureArray;
}

void SurfFeature::FreeFeature(float* array)
{
	if(array!=NULL)
	{
		delete[] array;
		array = NULL;
	}
}
