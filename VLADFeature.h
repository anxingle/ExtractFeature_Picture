#ifndef _VLADFEATURE_H
#define _VLADFEATURE_H


class VLADExtractor
{
public:
	VLADExtractor(float** vocabulary, int voc_rows, int voc_cols);
	~VLADExtractor();
	float* ExtractVLADFeature(float** feature, int feature_rows, int feature_cols);
	void FreeVLADFeature(float* array);
	int GetDimension(){return m_nDimension;}
	int GetClusterCenters(){return m_nClusterCenters;}
	int GetFeatureDim(){return m_nFeatureDim;}
private:
	int m_nDimension;              //��������ά��
	int m_nClusterCenters;  	  //�뱾����
	float** m_vocabulary;		//�뱾
	float** vlad_feature;       //�洢������vlad����
	float* Array_feature;       //��vlad_featureת��Ϊһά���鸳��Array_feature
	int m_nFeatureDim;          //Array_feature��ά����m_nClusterCenters*m_nDimension


	//����vlad����
	void computeFeature(float** feature, int feature_rows);
	//�����뱾m_vocabulary�о�������feature_i����ľ�������
	int getNearestCenterLabel(float* feature_i);
	//�ۼ�feature�;�������i�Ĳ�ֵ
	void accumulateVlad(float* feature, int center_i);
	//��������������ŷʽ����,�䳤��Ϊm_nDimension
	double euclidean(float *array1, float *array2);
	//��vlad_featureת��Ϊһά����Array_feature
	void toArray();
	//L2-norm
	void L2_Norm();
	//SSR_norm
	void SSR_Norm(float alpha);
	//uSSR-norm
	void uSSR_Norm(float alpha);

	//����ά��������ڴ�
	float **newArray(int row, int col);
	//���ٶ�ά����
	void deleteArray(float **Array, int row, int col);

};



#endif