#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Tensor.cuh"
#include "Display.hpp"
#include "../utils/ini.h"

class Transformer
{
	int embed_size;
	int vocab_size;
	int seq_len;
	int batch_size;
	int context_len;
	int num_seq;
	int xy_size;
	int head_size;
	float dropout_rate;
	float dropout_prob;
	
	vector<long long> token_ids;
	vector<long long> token_x;
	vector<long long> token_y;
	
	vector<vector<float>> embed_mat;
	vector<vector<float>> pos_mat;
	vector<vector<float>> embed_x;

	vector<vector<vector<float>>> X;
	vector<vector<vector<float>>> X_input;
	vector<vector<vector<int>>> dropout_mask;

	vector<vector<float>> wq;
	vector<vector<float>> wk;
	vector<vector<float>> wv;

	vector<vector<vector<float>>> q;
	vector<vector<vector<float>>> k;
	vector<vector<vector<float>>> v;

	vector<vector<vector<vector<float>>>> q_h;
	vector<vector<vector<vector<float>>>> k_h;
	vector<vector<vector<vector<float>>>> v_h;

	vector<float> gamma;
	vector<float> beta;	
public:
	
	Transformer(vector<long long>& ids)
	{
		mINI::INIFile file("../config.ini");
	    mINI::INIStructure in;

		if(file.read(in))
		{
			embed_size = stoi(in["GPT"]["Emdedding_size"]);
			vocab_size = stoi(in["GPT"]["Vocab_size"]);
			batch_size = stoi(in["GPT"]["Batch_size"]);
			seq_len = stoi(in["GPT"]["Seq_len"]);
			head_size = stoi(in["GPT"]["Head_size"]);
			dropout_rate = stof(in["GPT"]["Dropout_rate"]);
			dropout_prob = 1 - dropout_rate;
			token_ids = ids;
			context_len = token_ids.size();
			xy_size = context_len - 1;
			num_seq = xy_size - seq_len;
			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}

	void input_embedding()
	{
		token_x.reserve(xy_size);
		token_y.reserve(xy_size);

		for (int i = 0, j = 1; i < xy_size, j < context_len; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}

		embed_mat.reserve(vocab_size);
		pos_mat.reserve(xy_size);

		embed_mat = Tensor::random(vocab_size, embed_size);
		pos_mat = Tensor::random(xy_size, embed_size);

		embed_x.reserve(xy_size);
		
		for (int i = 0; i < xy_size; ++i) embed_x.push_back(Tensor::matadd(embed_mat[token_x[i]], pos_mat[i]));

		X.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			vector<vector<float>> temp;
			temp.reserve(seq_len + i);
			
			for (int j = 0 + i; j < seq_len + i; ++j) temp.push_back(embed_x[j]);
			X.push_back(temp);
		}
		dropout_mask.reserve(num_seq);
		dropout_mask = Tensor::dropout_mask(num_seq, seq_len, embed_size);
		
		X_input.reserve(num_seq);
		X_input = Tensor::dropout(X, dropout_mask, dropout_prob);
		
		gamma.reserve(embed_size);
		beta.reserve(embed_size);
		for (int i = 0; i < embed_size; ++i)
		{
			gamma.push_back(1.0f);
			beta.push_back(0.0f);
		}

		Tensor::layer_norm(X_input, gamma, beta);
	}

	void linear_projection()
	{
		wq.reserve(embed_size);
		wk.reserve(embed_size);
		wv.reserve(embed_size);

		wq = Tensor::random(embed_size, embed_size);
		wk = Tensor::random(embed_size, embed_size);
		wv = Tensor::random(embed_size, embed_size);

		q.reserve(num_seq);
		k.reserve(num_seq);
		v.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			q.push_back(Tensor::matmul(X_input[i], wq));
			k.push_back(Tensor::matmul(X_input[i], wk));
			v.push_back(Tensor::matmul(X_input[i], wv));
		}
		q_h.reserve(num_seq);
		k_h.reserve(num_seq);
		v_h.reserve(num_seq);

		q_h = Tensor::head_spliting(q, head_size);
		k_h = Tensor::head_spliting(k, head_size);
		v_h = Tensor::head_spliting(v, head_size);
	}
};
#endif