#ifndef _SURF_FEATURE_H_
#define _SURF_FEATURE_H_

#include "opencv2/opencv.hpp"
#include"opencv2/nonfree/features2d.hpp"  

class SurfFeature
{
public:
	void computeFeature(IplImage* image_color);
	float* SurfFeature::GetFeature();
	void SurfFeature::FreeFeature(float*);
private:
	static const int cwlDiameter = 10;  //提取surf特征时的窗口参数
	static const int cwlStep = 10; 
	static const int surfFeatureSize = 624 * 64;
	float* featureArray;
	cv::Mat GetDenseSurf(const cv::Mat& image);    //提取surf特征
	cv::Mat GetDenseSurf2(const cv::Mat& image);  //提取surf特征方法二
};
#endif