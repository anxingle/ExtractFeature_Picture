#ifndef _NEAREST_NEIGHBOR_H_
#define _NEAREST_NEIGHBOR_H_
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

void computeKNearestImage(vector<float*>& testFeature, 
						  unordered_map<string, vector<float*> >& trainFeatureMap,
						  int featureSize,
						  int K,
						  vector<pair<string, double> >& disMap);

#endif