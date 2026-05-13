#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Tensor.cuh"
#include "Display.hpp"
#include "Attension.cuh"
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
	int hidden_size;
	int block_size;
	float head_dim;
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
	vector<vector<vector<float>>> dropout_mask;
	

	vector<vector<float>> wq;
	vector<vector<float>> wk;
	vector<vector<float>> wv;
	vector<vector<float>> wo;
	vector<vector<float>> w1;
	vector<vector<float>> w2;	
	
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
			cout << "Parameters importing from config.ini....";
			embed_size = stoi(in["GPT"]["Emdedding_size"]);
			vocab_size = stoi(in["GPT"]["Vocab_size"]);
			batch_size = stoi(in["GPT"]["Batch_size"]);
			seq_len = stoi(in["GPT"]["Seq_len"]);
			dropout_rate = stof(in["GPT"]["Dropout_rate"]);
			dropout_prob = 1.0f - dropout_rate;
			head_size = stoi(in["GPT"]["Head_size"]);
			head_dim = embed_size / head_size;
			token_ids = ids;
			context_len = token_ids.size();
			xy_size = context_len - 1;
			num_seq = xy_size - seq_len;
			hidden_size = stoi(in["GPT"]["Hidden_size"]) * embed_size;
			block_size = stoi(in["GPT"]["Block_size"]);
			gamma.assign(embed_size, 1.0f);
			beta.assign(embed_size, 0.0f);
			cout << "Done....." << endl;
			
			cout << "Weigths Initializing.....";
			wq.reserve(embed_size);
			wk.reserve(embed_size);
			wv.reserve(embed_size);
			wo.reserve(embed_size);
			w1.reserve(embed_size);
			w1.reserve(hidden_size);
			wq = Tensor::weights(embed_size, embed_size);
			wk = Tensor::weights(embed_size, embed_size);
			wv = Tensor::weights(embed_size, embed_size);
			wo = Tensor::weights(embed_size, embed_size);
			w1 = Tensor::weights(embed_size, hidden_size);
			w2 = Tensor::weights(hidden_size, embed_size);
			cout << "Done....." << endl;
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

		embed_mat = Tensor::weights(vocab_size, embed_size);
		pos_mat = Tensor::weights(xy_size, embed_size);

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
	}

	void linear_projection(vector<vector<vector<float>>>& X_in)
	{
		q.clear();
		k.clear();
		v.clear();
		q_h.clear();
		k_h.clear();
		v_h.clear();

		q.reserve(num_seq);
		k.reserve(num_seq);
		v.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			q.push_back(Tensor::dot_product(X_in[i], wq));
			k.push_back(Tensor::dot_product(X_in[i], wk));
			v.push_back(Tensor::dot_product(X_in[i], wv));
		}
		
		q_h.reserve(num_seq);
		k_h.reserve(num_seq);
		v_h.reserve(num_seq);

		q_h = Tensor::head_spliting(q, head_size);
		k_h = Tensor::head_spliting(k, head_size);
		v_h = Tensor::head_spliting(v, head_size);

		for (int i = 0; i < num_seq; ++i)
		{
			q_h[i] = Tensor::transpose(q_h[i]);
			k_h[i] = Tensor::transpose(k_h[i]);
			v_h[i] = Tensor::transpose(v_h[i]);
		}
	}

	void forward_pass()
	{

		dropout_mask = Tensor::dropout_mask(num_seq, seq_len, embed_size, dropout_rate);
		X_input = Tensor::dropout(X, dropout_mask, dropout_prob);

		for (int i = 0; i < block_size; ++i)
		{
			cout << "=======================================================" << endl;
			cout << "Block " << i + 1 << " Begining...." << endl;
			cout << "=======================================================" << endl;
			
			Tensor::layer_norm(X_input, gamma, beta);

			cout << "Layer_norm - 1 Done...." << endl;
			Debug::shape(X_input);
			linear_projection(X_input);

			cout << "Linear_projection Done...." << endl;
			Debug::shape(q_h);
			Debug::shape(k_h);
			Debug::shape(v_h);
			auto attension_score = Attension::score(q_h, k_h, v_h, wo);

			cout << "Attension_score calculated Done...." << endl;

			dropout_mask = Tensor::dropout_mask(num_seq, seq_len, embed_size, dropout_rate);
			attension_score = Tensor::dropout(attension_score, dropout_mask, dropout_prob);

			cout << "Dropout - 1 Done....." << endl;

			attension_score = Tensor::matadd(attension_score, X);
	
			cout << "Residual - 1 Done...." << endl;

			auto mlp_output = attension_score;

			Tensor::layer_norm(mlp_output, gamma, beta);

			cout << "Layer_norm - 2 Done...." << endl;

			Linear::linear(mlp_output, w1, w2);
			
			cout << "Liner_Layer Done...." << endl;

			dropout_mask = Tensor::dropout_mask(num_seq, seq_len, embed_size, dropout_rate);
			mlp_output = Tensor::dropout(mlp_output, dropout_mask, dropout_prob);

			cout << "Dropout - 2 Done....." << endl;
			
			X_input = Tensor::matadd(attension_score, mlp_output);

			cout << "Residual - 2 Done...." << endl;
			
			X = X_input;

			cout << "Block " << i + 1 << " ended...." << endl;
			Debug::display(X_input[0]);
			cout << "=======================================================" << endl << endl << endl;

		}
	}
};

#endif