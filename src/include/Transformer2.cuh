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

	long long* Y;
	long long* TX;

	float* embed_x;
	float* embed_mat;
	float* pos_mat;

	float* w_vocab;
	float* wq;
	float* wk;
	float* wv;
	float* wo;
	float* w1;
	float* w2;
	float* q;
	float* k;
	float* v;

	float* q_h;
	float* k_h;
	float* v_h;	

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
			cudaMallocManaged(&embed_mat, (vocab_size * embed_size) * sizeof(float));
			cudaMallocManaged(&pos_mat, (seq_len * embed_size) * sizeof(float));
			cudaMallocManaged(&wq, (block_size * embed_size * embed_size) * sizeof(float));
			cudaMallocManaged(&wk, (block_size * embed_size * embed_size) * sizeof(float));
			cudaMallocManaged(&wv, (block_size * embed_size * embed_size) * sizeof(float));
			cudaMallocManaged(&wo, (block_size * embed_size * embed_size) * sizeof(float));
			
			cudaMallocManaged(&w1, (block_size * embed_size * hidden_size) * sizeof(float));
			cudaMallocManaged(&w2, (block_size * hidden_size * embed_size) * sizeof(float));
			
			cudaMallocManaged(&q, (block_size * seq_len * embed_size) * sizeof(float));
			cudaMallocManaged(&k, (block_size * seq_len * embed_size) * sizeof(float));
			cudaMallocManaged(&v, (block_size * seq_len * embed_size) * sizeof(float));
			
			cudaMallocManaged(&q_h, (block_size * seq_len * head_size * head_dim) * sizeof(float));
			cudaMallocManaged(&k_h, (block_size * seq_len * head_size * head_dim) * sizeof(float));
			cudaMallocManaged(&v_h, (block_size * seq_len * head_size * head_dim) * sizeof(float));

			cudaMallocManaged(&gamma1, (block_size * embed_size) * sizeof(float));
			cudaMallocManaged(&beta1, (block_size * embed_size) * sizeof(float));

			cudaMallocManaged(&gamma2, (block_size * embed_size) * sizeof(float));
			cudaMallocManaged(&beta2, (block_size * embed_size) * sizeof(float));

			cudaMallocManaged(&final_gamma, embed_size * sizeof(float));
			cudaMallocManaged(&final_beta, embed_size * sizeof(float));
			cout << "Done....." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}
};

#endif