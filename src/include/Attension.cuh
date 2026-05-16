#ifndef ATTENSION
#define ATTENSION

#include "Function.hpp"

using namespace std;

namespace Attension
{
	vector<vector<float>> score(const vector<vector<vector<float>>>& q,
								const vector<vector<vector<float>>>& k,
								const vector<vector<vector<float>>>& v, const vector<vector<float>>& wo)
	{
		int head_size = q.size();
		int seq_len = q[0].size();
		int head_dim = q[0][0].size();
		float scale = sqrt(head_dim);

		vector<vector<float>> attension_out;

		vector<vector<vector<float>>> attension;
		attension.reserve(head_size);

		for (int i = 0; i < head_size; ++i)
		{
			auto k_t = Tensor::transpose(k[i]);
			
			auto score = Tensor::dot_product(q[i], k_t);
			
			Tensor::casual_mask(score, scale);

			Function::softmax(score);

			attension.push_back(Tensor::dot_product(score, v[i]));
		}
		
		attension = Tensor::transpose(attension);
		for (int i = 0; i < seq_len; ++i) attension_out.push_back(Tensor::merge_head(attension[i]));

		attension_out = Tensor::dot_product(attension_out, wo);
		
		return attension_out;
	}
}

#endif