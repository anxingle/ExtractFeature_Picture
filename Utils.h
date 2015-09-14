#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <io.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


inline void getFiles( string path, vector<string>& files )
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( p.assign(path).append("\\").append(fileinfo.name), files );
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

inline string intToString(int i) {
	stringstream os;
	os << i;
	return os.str();
}

inline string splitFileName(string file) {
	size_t start = file.find_last_of("/\\") + 1;
	//size_t end = file.find_last_of(".");
	return file.substr(start);
}

inline float* newArray(int col) {
	float* array = new float[col];
	memset(array, 0, col*sizeof(float));
	return array;
}

inline void freeArray(float* array) {
	if (!array) delete[] array;
	array = NULL;
}

inline void freeVectorArray(vector<float*>& arrays) {
	int n = arrays.size();
	for (int i = 0; i < n; ++i) {
		delete[] arrays[i];
		arrays[i] = NULL;
	}
}

inline float** newMatrix(int row, int col)
{
	float **Array = new float *[row];
	for(int r = 0; r < row; r++)
	{
		Array[r] = new float[col];
		memset(Array[r], 0, col*sizeof(float));
	}
	return Array;
}

inline void freeMatrix(float** matrix, int row) {
	for (int r = 0; r < row; ++r) delete[] matrix[r];
	delete [] matrix;
	matrix = NULL;
}

inline void freeFeatureMap(unordered_map<string, vector<float*> >& imageMap) {
	for(auto& image : imageMap) {
		int n = image.second.size();
		for (int i = 0; i < n; ++i) {
			delete [] image.second[i];
			image.second[i] = NULL;
		}
	}
}

inline void loadFeatureBin(string trainFile, 
						   string featureFile, 
						   int featureSize,
						   unordered_map<string, vector<float*> >& featureMap) {
	ifstream imageInfo(trainFile.c_str());
	FILE* fid;
	fid = fopen(featureFile.c_str(), "rb");
	
	ofstream testOutfile("..\\..\\testDiff.txt");

	int imgSize;
	imageInfo >> imgSize;
	for (int i = 0; i < imgSize; ++i) {
		string imagePath;
		int proposalsNum;
		imageInfo >> imagePath >> proposalsNum;
		for (int r = 0; r < proposalsNum; ++r) {
			float* feature = newArray(featureSize);
			for (int c = 0; c < featureSize; ++c) {
				fread(&feature[c], sizeof(float),1, fid);
			}
			featureMap[imagePath].push_back(feature);
		}
	}
	testOutfile.close();
	imageInfo.close();
	fclose(fid);
}

inline void loadVocabulary( string file, float** matrix, int row, int col) {
	ifstream infile(file.c_str());
	if(!infile) {
		cout<<"open file failed: "<< file << endl;
		return;
	}
	for(int r = 0; r < row; r++) {
		for(int c = 0; c < col; c++) {
			infile >> matrix[r][c];
		}
	}
	infile.close();
}


inline void feature2Bin(float* feature, int featureSize, FILE* fin) {
	for (int i = 0; i < featureSize; ++i) {
		fwrite(feature + i, sizeof(float), 1, fin);
	}
}

inline void mat2Bin(const char* filename, const Mat& src) {
	FILE* fin;
	fin = fopen(filename, "wb");
	int row = src.rows;
	int col = src.cols;
	fwrite(&row, sizeof(int), 1, fin);
	fwrite(&col, sizeof(int), 1, fin);
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			fwrite(&(src.at<float>(r, c)), sizeof(float), 1, fin);
		}
	}
	fclose(fin);
}

inline void bin2Mat(const char* filename, Mat& dst) {
	FILE* fid;
	fid = fopen(filename, "rb");
	int row;
	int col;
	fread(&row, sizeof(int), 1, fid);
	fread(&col, sizeof(int), 1, fid);
	dst = Mat::zeros(row,col, CV_32F);
	for (int r = 0; r < row; ++r) {
		for (int c = 0; c < col; ++c) {
			fread(&(dst.at<float>(r, c)), sizeof(float), 1, fid);
		}
	}
	fclose(fid);
}

#endif