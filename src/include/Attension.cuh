#ifndef ATTENSION
#define ATTENSION

#include "Function.cuh"
using namespace std;

namespace Attension
{
	vector<vector<vector<float>>> score(vector<vector<vector<vector<float>>>>& q,
										vector<vector<vector<vector<float>>>>& k,
										vector<vector<vector<vector<float>>>>& v, vector<vector<float>>& wo)
	{
		int batch_size = q.size();
		int head_size = q[0].size();
		int seq_len = q[0][0].size();
		int head_dim = q[0][0][0].size();
		float scale = sqrt(head_dim);

		vector<vector<vector<float>>> attension_out;
		attension_out.reserve(batch_size);
		
		for (int i = 0; i < batch_size; ++i)
		{
			vector<vector<vector<float>>> attension;
			attension.reserve(head_size);
			for (int j = 0; j < head_size; ++j)
			{
				auto k_t = Tensor::transpose(k[i][j]);
				auto score = Tensor::dot_product(q[i][j], k_t);
				
				Tensor::casual_mask(score, scale);
				
				Function::softmax(score);

				attension.push_back(Tensor::dot_product(score, v[i][j]));
			}
			
			attension = Tensor::transpose(attension);
			
			vector<vector<float>> temp;
			for (int i = 0; i < seq_len; ++i) temp.push_back(Tensor::merge_head(attension[i]));
			
			attension_out.push_back(Tensor::matmul(temp, wo));
		}
		return attension_out;
	}
}

#endif