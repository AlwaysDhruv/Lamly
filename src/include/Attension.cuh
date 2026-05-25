#ifndef ATTENSION
#define ATTENSION

#include "Function.cuh"

using namespace std;

namespace Attension
{
	vector<vector<float>> score(const vector<vector<vector<float>>>& q,
								const vector<vector<vector<float>>>& k,
								const vector<vector<vector<float>>>& v,
								const vector<vector<float>>& wo,
								vector<vector<vector<float>>>& attension_score,
								vector<vector<vector<float>>>& scaled_score,
								vector<vector<vector<float>>>& masked_score,
								vector<vector<vector<float>>>& attension_prob,
								vector<vector<vector<float>>>& attension_out2,
								vector<vector<float>>& merged_heads)
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
			attension_score.push_back(score);
			
			vector<vector<float>> scaled = score;
			vector<vector<float>> masked = score;

			Tensor::casual_mask(score, scale, scaled, masked);
			scaled_score.push_back(scaled);
			masked_score.push_back(masked);

			Function::softmax(score);
			attension_prob.push_back(score);

			attension.push_back(Tensor::dot_product(score, v[i]));
		}
		attension_out2 = attension;
		
		attension = Tensor::transpose(attension);
		for (int i = 0; i < seq_len; ++i) attension_out.push_back(Tensor::merge_head(attension[i]));

		merged_heads = attension_out;

		attension_out = Tensor::dot_product(attension_out, wo);
		
		return attension_out;
	}
}

#endif