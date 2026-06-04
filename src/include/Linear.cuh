#ifndef LINEAR
#define LINEAR

#include "Function.cuh"

using namespace std;

namespace Linear
{
	void linear(vector<vector<vector<float>>>& X, vector<vector<float>>& w1, vector<vector<float>>& w2)
	{
		int batch_size = X.size();
		int seq_len = X[0].size();
		int embed_size = X[0][0].size();
		int hidden_size = w2.size();
		
		for (int i = 0; i < batch_size; ++i)
		{
			X[i] = Tensor::dot_product(X[i], w1);

			Function::gelu(X[i]);
			
			X[i] = Tensor::dot_product(X[i], w2);
		}
	}
}

#endif