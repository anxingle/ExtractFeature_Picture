#include "NearestNeighbors.h"
#include <algorithm>

bool comp (pair<string, double>& p1, pair<string, double>& p2) {
	return p1.second > p2.second;
}


//计算两个向量的欧式距离
double vecDistance(float *array1, float *array2, int len)
{
	double result = 0.0;
	for(int i = 0; i < len; i++)
	{
		result += (array1[i] - array2[i]) * (array1[i] - array2[i]);
	}
	return sqrt(result);
}

double computeDistance(vector<float*>& testFeature, vector<float*>& trainFeature, int featureSize) {
	double minDis = numeric_limits<double>::max();
	for (int i = 0; i < testFeature.size(); ++i) {
		for (int j = 0; j < trainFeature.size(); ++j) {
			double dis = vecDistance(testFeature[i], trainFeature[j], featureSize);
			if (dis < minDis) minDis = dis;
		}
	}
	return minDis;
}

void insertDis(string path, double dis, vector<pair<string, double> >& disMap) {
	disMap.push_back(make_pair(path, dis));
	int n = disMap.size();
	int cur = n - 1;
	for (int i = n - 2; i >= 0; --i) {
		if (disMap[i].second > dis) {
			swap(disMap[i], disMap[cur]);
			cur = i;
		} else {
			 break;
		}
	}
}

void computeKNearestImage(vector<float*>& testFeature, 
						  unordered_map<string, vector<float*> >& trainFeatureMap,
						  int featureSize,
						  int K,
						  vector<pair<string, double> >& disMap) {
	for (auto& trainImage : trainFeatureMap) {
		string trainImagePath = trainImage.first;
		vector<float* > trainFeature = trainImage.second;
		double dis = computeDistance(testFeature, trainFeature, featureSize);
		if (disMap.size() < K) {
			insertDis(trainImagePath, dis, disMap);
		} else {
			if (disMap[K - 1].second > dis) {
				disMap.pop_back();
				insertDis(trainImagePath, dis, disMap);
			}
		}
	}
}