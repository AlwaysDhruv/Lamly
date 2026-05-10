#ifndef ATTENSION
#define ATTENSION

using namespace std;

namespace Attension
{
	void score(vector<vector<vector<vector<float>>>>& q,
			vector<vector<vector<vector<float>>>>& k,
			vector<vector<vector<vector<float>>>>& v)
	{
		int batch_size = q.size();
		int head_size = q[0].size();
		
		float scale = sqrt(head_dim);

		for (int i = 0; i < batch_size; ++i)
		{
			for (int j = 0; j < head_size; ++j)
			{
				auto k_t = Tensor::transpose(k[i][j]);
				auto score = Tensor::dot_product(q[i][j], k_t);
				
				Tensor::casual_mask(score, scale);
				Tensor::softmax(score);
				
				auto attension_out = Tensor::dot_product(score, v[i][j]);

				Debug::shape(attension_out);
			}
			break;
		}
	}
}

#endif