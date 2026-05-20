#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Tensor.cuh"
#include "Display.hpp"
#include "Attension.cuh"
#include "Linear.cuh"
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
	int display;
	bool flag;

	vector<long long> token_ids;
	vector<long long> token_x;
	vector<long long> token_y;
	
	vector<vector<float>> embed_mat;
	vector<vector<float>> embed_mat_t;
	vector<vector<float>> pos_mat;
	vector<vector<float>> embed_x;

	vector<vector<float>> Y;
	vector<vector<vector<float>>> X;

	vector<vector<float>> wq;
	vector<vector<float>> wk;
	vector<vector<float>> wv;
	vector<vector<float>> wo;
	vector<vector<float>> w1;
	vector<vector<float>> w2;	
	
	vector<vector<float>> q;
	vector<vector<float>> k;
	vector<vector<float>> v;

	vector<vector<vector<float>>> q_h;
	vector<vector<vector<float>>> k_h;
	vector<vector<vector<float>>> v_h;	

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
			display = stoi(in["GPT"]["Display"]);
			flag = display==1 ? true : false;			
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
			wq = Tensor::random(embed_size, embed_size);
			wk = Tensor::random(embed_size, embed_size);
			wv = Tensor::random(embed_size, embed_size);
			wo = Tensor::random(embed_size, embed_size);
			w1 = Tensor::random(embed_size, hidden_size);
			w2 = Tensor::random(hidden_size, embed_size);
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

		embed_mat = Tensor::random(vocab_size, embed_size);
		pos_mat = Tensor::random(xy_size, embed_size);

		embed_x.reserve(xy_size);
		
		for (int i = 0; i < xy_size; ++i) embed_x.push_back(Tensor::matadd(embed_mat[token_x[i]], pos_mat[i]));

		X.reserve(num_seq);
		Y.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			vector<vector<float>> temp;
			vector<float> temp2;
			
			temp.reserve(seq_len + i);
			temp2.reserve(seq_len);

			for (int j = 0 + i; j < seq_len + i; ++j)
			{
				temp.push_back(embed_x[j]);
				temp2.push_back(token_y[j]);
			}
			X.push_back(temp);
			Y.push_back(temp2);
		}
	}

	void linear_projection(vector<vector<float>>& X_in)
	{
		q.clear();
		k.clear();
		v.clear();
		q_h.clear();
		k_h.clear();
		v_h.clear();

		q = Tensor::dot_product(X_in, wq);
		k = Tensor::dot_product(X_in, wk);
		v = Tensor::dot_product(X_in, wv);
		
		q_h.reserve(seq_len);
		k_h.reserve(seq_len);
		v_h.reserve(seq_len);

		q_h = Tensor::head_spliting(q, head_size);
		k_h = Tensor::head_spliting(k, head_size);
		v_h = Tensor::head_spliting(v, head_size);
		
		q_h = Tensor::transpose(q_h);
		k_h = Tensor::transpose(k_h);
		v_h = Tensor::transpose(v_h);
	}

	void Transformers()
	{
		int ct = 0;
		embed_mat_t = Tensor::transpose(embed_mat);
		for (int i = 0; i < num_seq; i+=batch_size)
		{
			flag ? cout << "======================================================================" << endl : cout << "";
			flag ? cout << "Batch " << ++ct << " Started...." << endl : cout << "";
			
			vector<vector<vector<float>>> X_input;
			vector<vector<vector<float>>> X_bnorm;
			vector<vector<vector<float>>> X_anorm;
			vector<vector<vector<float>>> loss_gradients;
			vector<vector<vector<float>>> hidden_states;
			vector<vector<float>> dw_vocab(embed_size, vector<float>(vocab_size, 0.0f));
			vector<vector<vector<float>>> dh;

			vector<vector<float>> mean;
			vector<vector<float>> var;
			vector<vector<float>> std;
			
			X_input.reserve(batch_size);
			
			loss_gradients.reserve(batch_size);
			hidden_states.reserve(batch_size);
			
			X_bnorm.reserve(batch_size);
			mean.reserve(batch_size);
			var.reserve(batch_size);
			std.reserve(batch_size);

			float loss = 0.0f;
			
			for (int seq = i, j = 0; seq < batch_size + i, j < batch_size; ++seq, ++j)
			{
				flag ? cout << "===========================================================" << endl : cout << "";
				flag ? cout << "seq " << seq + 1 << " Started...." << endl : cout << "";
				
				flag ? cout << "Input Dropouting......" : cout << "";
				auto dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
				auto X_input2 = Tensor::dropout(X[seq], dropout_mask, dropout_prob);
				flag ? cout << "Done......" << endl : cout << "";

				for (int i = 0; i < block_size; ++i)
				{
					flag ? cout << "=========================================" << endl : cout << "";
					flag ? cout << i + 1 << " Block started...." << endl : cout << "";
					flag ? cout << "=========================================" << endl : cout << "";
					auto residual = X_input2;
					
					flag ? cout << "Block First Layer normalizing....." : cout << "";
					Tensor::layer_norm(X_input2, gamma, beta);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Block Linear Projecting....." : cout << "";
					linear_projection(X_input2);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Block attension Score Calculating....." : cout << "";
					auto attension = Attension::score(q_h, k_h, v_h, wo);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Attension Score Dropouting......" : cout << "";
					dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
					X_input2 = Tensor::dropout(attension, dropout_mask, dropout_prob);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "First Residual Adding......" : cout << "";
					X_input2 = Tensor::matadd(residual, X_input2);
					flag ? cout << "Done..." << endl : cout << "";

					residual = X_input2;
					flag ? cout << "Block Second Layer normalizing....." : cout << "";
					Tensor::layer_norm(X_input2, gamma, beta);
					flag ? cout << "Done..." << endl << endl : cout << "";

					flag ? cout << "Linear Layer Started......" << endl : cout << "";
					
					flag ? cout << "Linear1 Calulating......" : cout << "";
					X_input2 = Tensor::dot_product(X_input2, w1);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Gelu Calculating......" : cout << "";
					Function::gelu(X_input2);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Linear2 Calulating......" : cout << "";
					X_input2 = Tensor::dot_product(X_input2, w2);
					flag ? cout << "Done..." << endl : cout << "";
					flag ? cout << "Linear Layer Calculated...." << endl << endl : cout << "";

					flag ? cout << "Linear Output Dropouting......" : cout << "";
					dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
					X_input2 = Tensor::dropout(attension, dropout_mask, dropout_prob);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Second Residual Adding......" : cout << "";
					X_input2 = Tensor::matadd(residual, X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "=========================================" << endl : cout << "";
					flag ? cout << i + 1 << " ended...." << endl : cout << "";
					flag ? cout << "=========================================" << endl << endl : cout << "";
				}	
				flag ? cout << "Final Layer normalizing....." : cout << "";
				X_bnorm.push_back(X_input2);
				auto [m, v, s, Xa] = Tensor::final_layer_norm(X_input2, gamma, beta);
				mean.push_back(m);
				var.push_back(v);
				std.push_back(s);
				X_anorm.push_back(Xa);
				hidden_states.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "LM Head Projecting....." : cout << "";
				X_input2 = Tensor::dot_product(X_input2, embed_mat_t);
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "Softmax....." : cout << "";
				Function::softmax(X_input2);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Calculating Loss....." : cout << "";
				for (int lss = 0; lss < seq_len; ++lss) loss += -log(X_input2[lss][Y[i][lss]]);
				X_input.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Calculating Gradients....." : cout << "";
				vector<vector<float>> gradient;
				for (int lss = 0; lss < seq_len; ++lss)
				{
					X_input2[lss][Y[i][lss]] -= 1;
					gradient.push_back(X_input2[lss]);
				}
				loss_gradients.push_back(gradient);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Done..." << endl << endl : cout << "";
				flag ? cout << "===========================================================" << endl : cout << "";
			}
			flag ? cout << "Calculating Batch " << ct << " Loss....." : cout << "";
			flag ? cout << "Done..." : cout << "";			
			flag ? cout << "Batch " << ct << " Loss : " << loss / (batch_size * seq_len) << endl : cout << "Batch " << ++ct << " Loss : " << loss / (batch_size * seq_len) << endl;

			flag ? cout << "LM head Backward....." : cout << "";
			dh.reserve(batch_size);
			for (int gra = 0; gra < batch_size; ++gra)
			{
				auto h_t = Tensor::transpose(hidden_states[gra]);
				auto sum = Tensor::dot_product(h_t, loss_gradients[gra]);
				dw_vocab = Tensor::matadd(dw_vocab, sum);
		
				auto embed_mat_t2 = Tensor::transpose(embed_mat_t);
				dh.push_back(Tensor::dot_product(loss_gradients[gra], embed_mat_t2));
			}
			flag ? cout << "Done..." : cout << "";

			flag ? cout << "Final Layer Norm Backward....." : cout << "";
			auto dbeta = Tensor::add(dh);
			auto dg = Tensor::matmul_e(dh, X_anorm);
			auto dgamma = Tensor::add(dg);
			flag ? cout << "Done..." << endl: cout << "";

			flag ? cout << "Batch " << ct << " ended...." << endl : cout << "";
			break;			
		}
	}
};

#endif