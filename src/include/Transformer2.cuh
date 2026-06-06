#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "Tensor.cuh"
#include "Display.hpp"
#include "../utils/ini.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

class Transformer
{
	int train;
	int display;
	int embed_size;
	int vocab_size;
	int seq_len;
	int batch_size;
	int tokens_size;
	int num_seq;
	int xy_size;
	int head_size;
	int hidden_size;
	int block_size;
	float head_dim;
	float scale;
	bool flag;
	float dropout_rate;
	float dropout_prob;
	float learning_rate;

	vector<long long> token_ids;
	vector<vector<long long>> x;
	vector<vector<long long>> y;

	float* embed_mat;
	float* pos_mat;

	float* wq;
	float* wk;
	float* wv;
	float* wo;
	float* w1;
	float* w2;
	float* w_vocab;

	float* gamma1;
	float* beta1;

	float* gamma2;
	float* beta2;

	float* final_gamma;
	float* final_beta;
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
			display = stoi(in["GPT"]["Display"]);
			flag = display==1 ? true : false;
			head_dim = embed_size / head_size;
			scale = sqrt(head_dim);
			token_ids = ids;
			tokens_size = token_ids.size();
			xy_size = tokens_size - 1;
			num_seq = xy_size - seq_len + 1;
			hidden_size = stoi(in["GPT"]["Hidden_size"]);
			block_size = stoi(in["GPT"]["Block_size"]);
			learning_rate = stof(in["GPT"]["Learning_rate"]);
			train = stoi(in["GPT"]["Train"]);
			cout << "Done....." << endl;
						
			cout << "Weigths Initializing....";

			Tensor::init_rng();

			size_t size = vocab_size * embed_size;
			cudaMalloc(&embed_mat, size  * sizeof(float));
			Tensor::random(embed_mat, size);

			size = seq_len * embed_size;
			cudaMalloc(&pos_mat, size * sizeof(float));
			Tensor::random(pos_mat, size);
			
			size = block_size * embed_size * embed_size;
			cudaMalloc(&wq, size * sizeof(float));
			cudaMalloc(&wk, size * sizeof(float));
			cudaMalloc(&wv, size * sizeof(float));
			cudaMalloc(&wo, size * sizeof(float));
			Tensor::random(wq, size);
			Tensor::random(wk, size);
			Tensor::random(wv, size);
			Tensor::random(wo, size);
			
			size = block_size * embed_size * hidden_size;
			cudaMalloc(&w1, size * sizeof(float));
			Tensor::random(w1, size);

			size = block_size * hidden_size * embed_size;
			cudaMalloc(&w2, size * sizeof(float));
			Tensor::random(w2, size);

			size = vocab_size * embed_size;
			cudaMalloc(&w_vocab, size  * sizeof(float));
			Tensor::random(w_vocab, size);
			
			size = block_size * embed_size;
			cudaMalloc(&gamma1, size * sizeof(float));
			cudaMalloc(&beta1, size * sizeof(float));
			cudaMalloc(&gamma2, (block_size * embed_size) * sizeof(float));
			cudaMalloc(&beta2, (block_size * embed_size) * sizeof(float));
			Tensor::fill(gamma1, 1.0f, size);
			Tensor::fill(beta1, 0.0f, size);
			Tensor::fill(gamma2, 1.0f, size);
			Tensor::fill(beta2, 0.0f, size);

			cudaMalloc(&final_gamma, embed_size * sizeof(float));
			cudaMalloc(&final_beta, embed_size * sizeof(float));
			Tensor::fill(final_gamma, 1.0f, embed_size);
			Tensor::fill(final_beta, 0.0f, embed_size);

			Tensor::destroy_rng();
			
			cout << "Done....." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}

	void input_embedding()
	{
		vector<long long> token_x;
		vector<long long> token_y;

		token_x.reserve(tokens_size - 1);
		token_y.reserve(tokens_size);

		for (int i = 0, j = 1; i < tokens_size - 1, j < tokens_size; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}
		
		x.reserve(num_seq);
		y.reserve(num_seq);
		
		for (int i = 0; i < num_seq; ++i)
		{
			vector<long long> temp1;
			vector<long long> temp2;
			temp1.reserve(seq_len);
			temp2.reserve(seq_len);
			for (int j = 0 + i; j < seq_len + i; ++j)
			{
				temp1.push_back(token_x[j]);
				temp2.push_back(token_y[j]);
			}
			x.push_back(temp1);
			y.push_back(temp2);
		}
	}

	void prepare()
	{
		for(int i = 0; i < num_seq; i+=batch_size)
		{
			size_t current_batch_size = num_seq < i + batch_size ?  (num_seq - i) + i : batch_size + i;

			long long* token_x_gpu;
			size_t size = current_batch_size * seq_len;
			cudaMalloc(&x_input, size * sizeof(long long));
			
			for(int j = i; j < current_batch_size; j++)
			{
				for(int k = j; k < seq_len + j; k++)
				{
					//copy tokens to gpu
				}
			}
		}
	}
};

#endif