#ifndef _NEAREST_NEIGHBOR_H_
#define _NEAREST_NEIGHBOR_H_
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

void computeKNearestImage(vector<pair<string, float*> >& testFeature, 
						  int featureSize,
						  unordered_map<string, vector<pair<string, float*> > >& trainFeatureMap,
						  unordered_map<string, string>& trainImageMap,
						  int K,
						  vector<pair<string, double> >& disMap);

#endif