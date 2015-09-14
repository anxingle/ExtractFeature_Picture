#include <string>
#include <fstream>
#include <sstream>
#include "SurfFeature.h"
using namespace std; //why


SurfFeature::SurfFeature()
{
	
}

SurfFeature::~SurfFeature()
{

}


void SurfFeature::computeFeature(IplImage *image_color)
{
	//ת��Ϊ�Ҷ�ͼ��
	IplImage *GrayImage = cvCreateImage(cvGetSize(image_color),IPL_DEPTH_8U,1);
	cvCvtColor(image_color,GrayImage,CV_RGB2GRAY);
	cv::Mat image(GrayImage);

	//��ȡ������ת��Ϊһά����
	cv::Mat surfFeature = GetDenseSurf2(image);
	surfFeature.reshape(0,1);
	assert(surfFeatureSize == surfFeature.total());
	featureArray = new float[surfFeatureSize];
	memcpy(featureArray, surfFeature.ptr<float>(0), surfFeatureSize*sizeof(float));

	//�ͷŻҶ�ͼ��
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


float* SurfFeature::GetFeature()
{
	return featureArray;
}

float** SurfFeature::GetFeature(const cv::Mat& image)
{
	GetDenseSurf(image);
	Mat2Array();
	return SurfArray;
}
int SurfFeature::GetRows() const
{
	return rows;
}
int SurfFeature::GetCols() const
{
	return cols;
}

void SurfFeature::GetDenseSurf(const cv::Mat& image)
{
	int imageRows = image.rows;
	int imageCols = image.cols;
	if( (imageRows < cwlDiameter) || (imageCols < cwlDiameter) )
	{
		return;
	}

	int row_begin = cwlStep;
	int row_end = imageRows - cwlStep + 1;
	int col_begin = cwlStep;
	int col_end = imageCols - cwlStep + 1;
	
	std::vector<cv::KeyPoint> denseKeyPoints;
	
	for(int r = row_begin; r <= row_end; r += cwlStep)
	{
		for(int c = col_begin; c <= col_end; c += cwlStep)
		{
			denseKeyPoints.push_back(cv::KeyPoint(c, r, cwlDiameter, 0));
		}
	}

	//int minHessian = 400;
	//cv::SurfFeatureDetector detector( minHessian );
	//detector.detect(image, denseKeyPoints);
	cv::SurfDescriptorExtractor extractor;
	extractor.compute(image,denseKeyPoints,SurfMat);
}

void SurfFeature::Mat2Array()
{
	//ͳ�Ʒ�����
	std::vector<int> nonZeroRow;
	int countNonZero= 0;
	for(int r = 0; r < SurfMat.rows; r++)
	{
		int num_zero = cv::countNonZero(SurfMat.row(r));
		if(num_zero != 0)
		{
			countNonZero++;
			nonZeroRow.push_back(r);
		}
	}

	//����feature��ά��
	rows = countNonZero;
	cols = SurfMat.cols;

	//������ת��Ϊ����
	SurfArray = newArray();
	for(size_t i = 0; i < nonZeroRow.size(); i++)
	{
		memcpy(SurfArray[i], SurfMat.ptr<float>(nonZeroRow[i]), cols*sizeof(float));

	}
}

//����ά��������ڴ�
float** SurfFeature::newArray()
{
	float **Array = new float *[rows];
	for(int r = 0; r < rows; r++)
	{
		Array[r] = new float[cols];
		memset(Array[r], 0, cols*sizeof(float));
	}
	return Array;
}

//���ٶ�ά����
void SurfFeature::FreeFeature(float **Array)
{
	for(int r = 0; r < rows; r++)
	{
		delete [cols]Array[r];
		Array[r] = NULL;
	}
	delete [rows]Array;
	Array = NULL;
}
