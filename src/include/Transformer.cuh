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

	vector<vector<long long>> Y;
	vector<vector<long long>> TX;


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
			w2.reserve(hidden_size);
			embed_mat.reserve(vocab_size);
			pos_mat.reserve(seq_len);

			wq = Tensor::random(embed_size, embed_size);
			wk = Tensor::random(embed_size, embed_size);
			wv = Tensor::random(embed_size, embed_size);
			wo = Tensor::random(embed_size, embed_size);
			w1 = Tensor::random(embed_size, hidden_size);
			w2 = Tensor::random(hidden_size, embed_size);
			embed_mat = Tensor::random(vocab_size, embed_size);
			pos_mat = Tensor::random(seq_len, embed_size);			
			cout << "Done....." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}

	void input_embedding()
	{
		vector<long long> token_x;
		vector<long long> token_y;

		token_x.reserve(xy_size);
		token_y.reserve(xy_size);

		for (int i = 0, j = 1; i < xy_size, j < context_len; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}

		TX.reserve(num_seq);
		Y.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			vector<long long> temp;
			vector<long long> temp2;
		
			temp.reserve(seq_len + i);
			temp2.reserve(seq_len);

			for (int j = 0 + i; j < seq_len + i; ++j)
			{
				temp.push_back(token_x[j]);
				temp2.push_back(token_y[j]);
			}
			TX.push_back(temp);
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
		embed_mat_t = Tensor::transpose2(embed_mat);

		for (int batch = 0; batch < num_seq; batch+=batch_size)
		{
			flag ? cout << "======================================================================" << endl : cout << "";
			flag ? cout << "Batch " << ++ct << " Started...." << endl << endl : cout << "";
			
			vector<vector<vector<float>>> X;
			X.reserve(batch_size);
			for (int seq = batch; seq < batch_size + batch; ++seq)
			{
				vector<vector<float>> temp;
				temp.reserve(seq_len);
				for (int i = 0; i < seq_len; ++i)
				{
					temp.push_back(Tensor::matadd(embed_mat[TX[seq][i]], pos_mat[i]));
				}
				X.push_back(temp);
			}

			vector<vector<vector<float>>> X_input;
			vector<vector<vector<float>>> loss_gradients;
			vector<vector<vector<float>>> hidden_states;
			vector<vector<float>> dw_vocab(embed_size, vector<float>(vocab_size, 0.0f));
			vector<vector<vector<float>>> dy;
			vector<vector<vector<vector<float>>>> dropout_mask_mlp;
			vector<vector<vector<vector<float>>>> gelu_output;

			vector<vector<vector<vector<float>>>> l1_X_norm;
			vector<vector<vector<vector<float>>>> l1_X_meaned;
			vector<vector<vector<float>>> l1_mean;
			vector<vector<vector<float>>> l1_var;
			vector<vector<vector<float>>> l1_std;

			vector<vector<vector<vector<float>>>> l2_X_norm;
			vector<vector<vector<vector<float>>>> l2_X_meaned;
			vector<vector<vector<float>>> l2_mean;
			vector<vector<vector<float>>> l2_var;
			vector<vector<vector<float>>> l2_std;

			vector<vector<vector<float>>> final_X_norm;
			vector<vector<vector<float>>> final_X_meaned;
			vector<vector<float>> final_mean;
			vector<vector<float>> final_var;
			vector<vector<float>> final_std;
			
			vector<vector<float>> dw2(hidden_size, vector<float>(embed_size, 0.0f));

			X_input.reserve(batch_size);
			
			loss_gradients.reserve(batch_size);
			hidden_states.reserve(batch_size);
			
			dy.reserve(batch_size);
			dropout_mask_mlp.reserve(batch_size);
			gelu_output.reserve(batch_size);

			l1_X_norm.reserve(batch_size);
			l1_X_meaned.reserve(batch_size);
			l1_mean.reserve(batch_size);
			l1_var.reserve(batch_size);
			l1_std.reserve(batch_size);

			l2_X_norm.reserve(batch_size);
			l2_X_meaned.reserve(batch_size);
			l2_mean.reserve(batch_size);
			l2_var.reserve(batch_size);
			l2_std.reserve(batch_size);

			final_X_norm.reserve(batch_size);
			final_X_meaned.reserve(batch_size);
			final_mean.reserve(batch_size);
			final_var.reserve(batch_size);
			final_std.reserve(batch_size);
				
			float loss = 0.0f;
			flag ? cout << "Batch " << ct << "Forward pass Started...." << endl : cout << "";
			flag ? cout << "================================================================" << endl : cout << "";
			for (int seq = 0; seq < batch_size; ++seq)
			{
				vector<vector<vector<float>>> temp;
				vector<vector<vector<float>>> temp1;

				vector<vector<vector<float>>> t_l1_X_norm;
				vector<vector<vector<float>>> t_l1_X_meaned;
				vector<vector<float>> t_l1_mean;
				vector<vector<float>> t_l1_var;
				vector<vector<float>> t_l1_std;

				vector<vector<vector<float>>> t_l2_X_norm;
				vector<vector<vector<float>>> t_l2_X_meaned;
				vector<vector<float>> t_l2_mean;
				vector<vector<float>> t_l2_var;
				vector<vector<float>> t_l2_std;
				
				t_l1_X_norm.reserve(block_size);
				t_l1_X_meaned.reserve(block_size);
				t_l1_mean.reserve(block_size);
				t_l1_var.reserve(block_size);
				t_l1_std.reserve(block_size);

				t_l2_X_norm.reserve(block_size);
				t_l2_X_meaned.reserve(block_size);
				t_l2_mean.reserve(block_size);
				t_l2_var.reserve(block_size);
				t_l2_std.reserve(block_size);
			
				flag ? cout << "=======================================================" << endl : cout << "";
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
					
					t_l1_X_norm.push_back(X_input2);
					
					vector<vector<float>> t2_l1_X_meaned;
					vector<float> t2_l1_mean;
					vector<float> t2_l1_var;
					vector<float> t2_l1_std;
					
					t2_l1_X_meaned.reserve(seq_len);
					t2_l1_mean.reserve(seq_len);
					t2_l1_var.reserve(seq_len);
					t2_l1_std.reserve(seq_len);

					Tensor::layer_norm(X_input2, gamma, beta, t2_l1_mean, t2_l1_var, t2_l1_std, t2_l1_X_meaned);

					t_l1_mean.push_back(t2_l1_mean);
					t_l1_var.push_back(t2_l1_var);
					t_l1_std.push_back(t2_l1_std);
					t_l1_X_meaned.push_back(t2_l1_X_meaned);

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
					t_l2_X_norm.push_back(X_input2);
					
					vector<vector<float>> t2_l2_X_meaned;
					vector<float> t2_l2_mean;
					vector<float> t2_l2_var;
					vector<float> t2_l2_std;
					
					t2_l2_X_meaned.reserve(seq_len);
					t2_l2_mean.reserve(seq_len);
					t2_l2_var.reserve(seq_len);
					t2_l2_std.reserve(seq_len);

					Tensor::layer_norm(X_input2, gamma, beta, t2_l2_mean, t2_l2_var, t2_l2_std, t2_l2_X_meaned);
					
					t_l2_mean.push_back(t2_l2_mean);
					t_l2_var.push_back(t2_l2_var);
					t_l2_std.push_back(t2_l2_std);
					t_l2_X_meaned.push_back(t2_l2_X_meaned);
					flag ? cout << "Done..." << endl << endl : cout << "";

					flag ? cout << "Linear Layer Started......" << endl : cout << "";
					
					flag ? cout << "Linear1 Calulating......" : cout << "";
					X_input2 = Tensor::dot_product(X_input2, w1);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Gelu Calculating......" : cout << "";
					Function::gelu(X_input2);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Linear2 Calulating......" : cout << "";
					temp.push_back(X_input2);
					X_input2 = Tensor::dot_product(X_input2, w2);
					flag ? cout << "Done..." << endl : cout << "";
					flag ? cout << "Linear Layer Calculated...." << endl << endl : cout << "";

					flag ? cout << "Linear Output Dropouting......" : cout << "";
					dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
					temp1.push_back(dropout_mask);
					X_input2 = Tensor::dropout(attension, dropout_mask, dropout_prob);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Second Residual Adding......" : cout << "";
					X_input2 = Tensor::matadd(residual, X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "=========================================" << endl : cout << "";
					flag ? cout << i + 1 << " ended...." << endl : cout << "";
					flag ? cout << "=========================================" << endl << endl : cout << "";
				}
				dropout_mask_mlp.push_back(temp1);
				gelu_output.push_back(temp);
				
				l1_X_norm.push_back(t_l1_X_norm);
				l1_mean.push_back(t_l1_mean);
				l1_var.push_back(t_l1_var);
				l1_std.push_back(t_l1_std);
				l1_X_meaned.push_back(t_l1_X_meaned);

				l2_X_norm.push_back(t_l2_X_norm);
				l2_mean.push_back(t_l2_mean);
				l2_var.push_back(t_l2_var);
				l2_std.push_back(t_l2_std);
				l2_X_meaned.push_back(t_l2_X_meaned);

				flag ? cout << "Final Layer normalizing....." : cout << "";
	
				vector<vector<float>> t2_final_X_meaned;
				vector<float> t2_final_mean;
				vector<float> t2_final_var;
				vector<float> t2_final_std;

				final_X_norm.push_back(X_input2);
				Tensor::layer_norm(X_input2, gamma, beta, t2_final_mean, t2_final_var, t2_final_std, t2_final_X_meaned);
				
				final_mean.push_back(t2_final_mean);
				final_var.push_back(t2_final_var);
				final_std.push_back(t2_final_std);
				final_X_meaned.push_back(t2_final_X_meaned);

				hidden_states.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "LM Head Projecting....." : cout << "";
				X_input2 = Tensor::dot_product(X_input2, embed_mat_t);
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "Softmax....." : cout << "";
				Function::softmax(X_input2);
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "Calculating Loss....." : cout << "";
				for (int lss = 0; lss < seq_len; ++lss) loss += -log(X_input2[lss][Y[batch][lss]]);
				X_input.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Calculating Gradients....." : cout << "";
				vector<vector<float>> gradient;
				for (int lss = 0; lss < seq_len; ++lss)
				{
					X_input2[lss][Y[batch][lss]] -= 1.0f;
					gradient.push_back(X_input2[lss]);
				}
				loss_gradients.push_back(gradient);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Done..." << endl << endl : cout << "";
				flag ? cout << "=======================================================" << endl : cout << "";
			}
			flag ? cout << "Calculating Batch " << ct << " Loss....." : cout << "";
			flag ? cout << "Done..." : cout << "";
			flag ? cout << "Batch " << ct << " Loss : " << loss / (batch_size * seq_len) << endl : cout << "Batch " << ++ct << " Loss : " << loss / (batch_size * seq_len) << endl;
			flag ? cout << "Batch " << ct << " Forward pass ended...." << endl : cout << "";
			flag ? cout << "================================================================" << endl << endl : cout << "";

			flag ? cout << "Batch " << ct << " Backward pass Started...." << endl : cout << "";
			flag ? cout << "================================================================" << endl << endl : cout << "";

			flag ? cout << "LM head Backward....." : cout << "";
			for (int gra = 0; gra < batch_size; ++gra)
			{
				auto h_t = Tensor::transpose(hidden_states[gra]);
				auto sum = Tensor::dot_product(h_t, loss_gradients[gra]);
				dw_vocab = Tensor::matadd(dw_vocab, sum);
	
				auto embed_mat_t2 = Tensor::transpose2(embed_mat_t);
				dy.push_back(Tensor::dot_product(loss_gradients[gra], embed_mat_t2));
			}
			flag ? cout << "Done..." << endl: cout << "";

			flag ? cout << "Final Layer Norm Backward....." : cout << "";
			auto dbeta = Tensor::add(dy);
			auto dg = Tensor::matmul_e(dy, final_X_meaned);
			auto dgamma = Tensor::add(dg);
			auto dx_hat = Tensor::normalized_gradient(dy, gamma);
			auto dvar = Tensor::variance_gradient(final_X_norm, dx_hat, final_mean, final_var);
			auto dmean = Tensor::mean_gradient(dx_hat, final_X_meaned, dvar, final_std);
			dy = Tensor::input_gradient(dx_hat, final_X_meaned, dvar, dmean, final_std);
			flag ? cout << "Done..." << endl: cout << "";

			flag ? cout << "Transfomer Blocks Backwarding Started....." : cout << "";
			
			flag ? cout << "Done..." << endl: cout << "";

			flag ? cout << "Batch " << ct << "Backward pass ended...." << endl : cout << "";
			flag ? cout << "================================================================" << endl << endl : cout << "";

			flag ? cout << "Batch " << ct << " ended...." << endl : cout << "";
		}
	}
};

#endif