#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <io.h>
using namespace std;


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

inline string splitFileName(const string& file) {
	size_t found1 = file.find_last_of("/\\");
	size_t found2 = file.find_last_of("_");
	return file.substr(found1 + 1, found2 - found1 - 1);
}

inline void LoadFeature(string filename, 
						int featureSize, 
						unordered_map<string, vector<pair<string, float*> > >& trainFeatureMap,
						unordered_map<string, string>& trainImageMap) {
	ifstream infile(filename.c_str());
	if (!infile) {
		cerr << "Open file failed!" << filename << endl;
		return;
	}
	string imagePath;
	while ((infile >> imagePath)) {
		cout << imagePath << endl;
		float* feature = new float[featureSize];
		for (int i = 0; i < featureSize; ++i) infile >> feature[i];
		string imageName = splitFileName(imagePath);
		trainFeatureMap[imageName].push_back(make_pair(imagePath, feature));
		if (trainImageMap.count(imageName) == 0) {
			string prefixPath = imagePath.substr(0, imagePath.find_last_of("_"));
			string suffixPath = imagePath.substr(imagePath.find_last_of("."));
			trainImageMap[imageName] = prefixPath + "_0" + suffixPath;
		}
	}
}

inline void freeFeatureMap(unordered_map<string, vector<pair<string, float*> > >& imageMap) {
	for(auto& image : imageMap) {
		int n = image.second.size();
		for (int i = 0; i < n; ++i) {
			delete [] image.second[i].second;
		}
	}
}

inline string intToString(int i) {
	stringstream os;
	os << i;
	return os.str();
}

#endif