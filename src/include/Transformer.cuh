#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include <cmath>
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
	int display;
	bool flag;

	vector<long long> token_ids;

	vector<vector<float>> embed_mat;
	vector<vector<float>> pos_mat;
	vector<vector<float>> embed_x;

	vector<vector<long long>> Y;
	vector<vector<long long>> TX;

	vector<vector<float>> w_vocab;
	vector<vector<vector<float>>> wq;
	vector<vector<vector<float>>> wk;
	vector<vector<vector<float>>> wv;
	vector<vector<vector<float>>> wo;
	vector<vector<vector<float>>> w1;
	vector<vector<vector<float>>> w2;
	vector<vector<float>> q;
	vector<vector<float>> k;
	vector<vector<float>> v;

	vector<vector<vector<float>>> q_h;
	vector<vector<vector<float>>> k_h;
	vector<vector<vector<float>>> v_h;	

	vector<vector<float>> gamma1;
	vector<vector<float>> beta1;

	vector<vector<float>> gamma2;
	vector<vector<float>> beta2;

	vector<float> final_gamma;
	vector<float> final_beta;
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
			hidden_size = stoi(in["GPT"]["Hidden_size"]);
			block_size = stoi(in["GPT"]["Block_size"]);
			cout << "Done....." << endl;

			cout << "Weigths Initializing....";
			embed_mat.reserve(vocab_size);
			pos_mat.reserve(seq_len);
			wq.reserve(block_size);
			wk.reserve(block_size);
			wv.reserve(block_size);
			wo.reserve(block_size);
			w1.reserve(block_size);
			w2.reserve(block_size);	
			
			embed_mat = Tensor::random(vocab_size, embed_size);
			pos_mat = Tensor::random(seq_len, embed_size);
			gamma1.assign(block_size, vector<float>(embed_size, 1.0f));
			beta1.assign(block_size, vector<float>(embed_size, 0.0f));
			gamma2.assign(block_size, vector<float>(embed_size, 1.0f));
			beta2.assign(block_size, vector<float>(embed_size, 0.0f));			
			final_gamma.assign(embed_size, 1.0f);
			final_beta.assign(embed_size, 0.0f);
			wq = Tensor::random(block_size, embed_size, embed_size);
			wk = Tensor::random(block_size, embed_size, embed_size);
			wv = Tensor::random(block_size, embed_size, embed_size);
			wo = Tensor::random(block_size, embed_size, embed_size);
			w1 = Tensor::random(block_size, embed_size, hidden_size);
			w2 = Tensor::random(block_size, hidden_size, embed_size);
			w_vocab = Tensor::transpose(embed_mat);
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

	void linear_projection(vector<vector<float>>& X_in, int index)
	{
		q.clear();
		k.clear();
		v.clear();
		q_h.clear();
		k_h.clear();
		v_h.clear();

		q = Tensor::dot_product(X_in, wq[index]);
		k = Tensor::dot_product(X_in, wk[index]);
		v = Tensor::dot_product(X_in, wv[index]);
		
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
				for (int i = 0; i < seq_len; ++i) temp.push_back(Tensor::matadd(embed_mat[TX[seq][i]], pos_mat[i]));
				X.push_back(temp);
			}

			vector<vector<vector<float>>> X_input;

			vector<vector<vector<vector<float>>>> l1_X;
			vector<vector<vector<vector<float>>>> l1_X_STD;
			vector<vector<vector<float>>> l1_mean;
			vector<vector<vector<float>>> l1_var;
			vector<vector<vector<float>>> l1_std;

			vector<vector<vector<vector<float>>>> l2_X;
			vector<vector<vector<vector<float>>>> l2_X_STD;
			vector<vector<vector<float>>> l2_mean;
			vector<vector<vector<float>>> l2_var;
			vector<vector<vector<float>>> l2_std;

			vector<vector<vector<float>>> final_X;
			vector<vector<vector<float>>> final_X_STD;
			vector<vector<float>> final_mean;
			vector<vector<float>> final_var;
			vector<vector<float>> final_std;

			vector<vector<vector<vector<float>>>> QKV_input;
			vector<vector<vector<vector<float>>>> Q_cache;
			vector<vector<vector<vector<float>>>> K_cache;
			vector<vector<vector<vector<float>>>> V_cache;
			vector<vector<vector<vector<vector<float>>>>> Q_cache_H;
			vector<vector<vector<vector<vector<float>>>>> K_cache_H;
			vector<vector<vector<vector<vector<float>>>>> V_cache_H;

			vector<vector<vector<vector<vector<float>>>>> attension_score;
			vector<vector<vector<vector<vector<float>>>>> scaled_score;
			vector<vector<vector<vector<vector<float>>>>> masked_score;
			vector<vector<vector<vector<vector<float>>>>> attension_prob;
			vector<vector<vector<vector<vector<float>>>>> attension_out;
			vector<vector<vector<vector<float>>>> merged_heads;
			vector<vector<vector<vector<float>>>> attension_projected;
			vector<vector<vector<vector<float>>>> attension_mask;
			vector<vector<vector<vector<float>>>> attension_residual;

			vector<vector<vector<vector<float>>>> gelu_input;
			vector<vector<vector<vector<float>>>> gelu_output;
			vector<vector<vector<vector<float>>>> mlp_mask;

			vector<vector<vector<vector<float>>>> linear1_input;
			vector<vector<vector<vector<float>>>> linear1_output;

			vector<vector<vector<vector<float>>>> linear2_input;
			vector<vector<vector<vector<float>>>> linear2_output;

			vector<vector<vector<vector<float>>>> mlp_residual;

			vector<vector<vector<float>>> logits;

			vector<vector<vector<float>>> softmax_probs;

			vector<vector<vector<float>>> d_logits;
			vector<vector<vector<float>>> hidden_states;

			vector<vector<long long>> target_ids;

			X_input.reserve(batch_size);			

			l1_X.reserve(batch_size);
			l1_X_STD.reserve(batch_size);
			l1_mean.reserve(batch_size);
			l1_var.reserve(batch_size);
			l1_std.reserve(batch_size);

			l2_X.reserve(batch_size);
			l2_X_STD.reserve(batch_size);
			l2_mean.reserve(batch_size);
			l2_var.reserve(batch_size);
			l2_std.reserve(batch_size);

			final_X.reserve(batch_size);
			final_X_STD.reserve(batch_size);
			final_mean.reserve(batch_size);
			final_var.reserve(batch_size);
			final_std.reserve(batch_size);
				
			QKV_input.reserve(batch_size);
			Q_cache.reserve(batch_size);
			K_cache.reserve(batch_size);
			V_cache.reserve(batch_size);
			Q_cache_H.reserve(batch_size);
			K_cache_H.reserve(batch_size);
			V_cache_H.reserve(batch_size);

			attension_score.reserve(batch_size);
			scaled_score.reserve(batch_size);
			masked_score.reserve(batch_size);
			attension_prob.reserve(batch_size);
			attension_out.reserve(batch_size);
			merged_heads.reserve(batch_size);
			attension_projected.reserve(batch_size);
			attension_mask.reserve(batch_size);
			attension_residual.reserve(batch_size);

			gelu_input.reserve(batch_size);
			gelu_output.reserve(batch_size);
			mlp_mask.reserve(batch_size);

			linear1_input.reserve(batch_size);
			linear1_output.reserve(batch_size);

			linear2_input.reserve(batch_size);
			linear2_output.reserve(batch_size);

			mlp_residual.reserve(batch_size);
			
			logits.reserve(batch_size);

			softmax_probs.reserve(batch_size);

			d_logits.reserve(batch_size);
			hidden_states.reserve(batch_size);

			target_ids.reserve(batch_size);

			float loss = 0.0f;
			
			flag ? cout << "Batch " << ct << "Forward pass Started...." << endl : cout << "";
			flag ? cout << "================================================================" << endl : cout << "";
			
			for (int seq = 0; seq < batch_size; ++seq)
			{

				vector<vector<vector<float>>> t_l1_X;
				vector<vector<vector<float>>> t_l1_X_STD;
				vector<vector<float>> t_l1_mean;
				vector<vector<float>> t_l1_var;
				vector<vector<float>> t_l1_std;

				vector<vector<vector<float>>> t_l2_X;
				vector<vector<vector<float>>> t_l2_X_STD;
				vector<vector<float>> t_l2_mean;
				vector<vector<float>> t_l2_var;
				vector<vector<float>> t_l2_std;

				vector<vector<vector<float>>> t_QKV_input;
				vector<vector<vector<float>>> t_Q_cache;
				vector<vector<vector<float>>> t_K_cache;
				vector<vector<vector<float>>> t_V_cache;
				vector<vector<vector<vector<float>>>> t_Q_cache_H;
				vector<vector<vector<vector<float>>>> t_K_cache_H;
				vector<vector<vector<vector<float>>>> t_V_cache_H;
				
				vector<vector<vector<vector<float>>>> t_attension_score;
				vector<vector<vector<vector<float>>>> t_scaled_score;
				vector<vector<vector<vector<float>>>> t_masked_score;
				vector<vector<vector<vector<float>>>> t_attension_prob;
				vector<vector<vector<vector<float>>>> t_attension_out;
				vector<vector<vector<float>>> t_merged_heads;
				vector<vector<vector<float>>> t_attension_projected;
				vector<vector<vector<float>>> t_attension_mask;
				vector<vector<vector<float>>> t_attension_residual;

				vector<vector<vector<float>>> t_gelu_input;
				vector<vector<vector<float>>> t_gelu_output;
				vector<vector<vector<float>>> t_mlp_mask;

				vector<vector<vector<float>>> t_linear1_input;
				vector<vector<vector<float>>> t_linear1_output;

				vector<vector<vector<float>>> t_linear2_input;
				vector<vector<vector<float>>> t_linear2_output;

				vector<vector<vector<float>>> t_mlp_residual;

				t_l1_X.reserve(block_size);
				t_l1_X_STD.reserve(block_size);
				t_l1_mean.reserve(block_size);
				t_l1_var.reserve(block_size);
				t_l1_std.reserve(block_size);

				t_l2_X.reserve(block_size);
				t_l2_X_STD.reserve(block_size);
				t_l2_mean.reserve(block_size);
				t_l2_var.reserve(block_size);
				t_l2_std.reserve(block_size);
			
				t_QKV_input.reserve(block_size);
				t_Q_cache.reserve(block_size);
				t_K_cache.reserve(block_size);
				t_V_cache.reserve(block_size);
				t_Q_cache_H.reserve(block_size);
				t_K_cache_H.reserve(block_size);
				t_V_cache_H.reserve(block_size);

				t_attension_score.reserve(block_size);
				t_scaled_score.reserve(block_size);
				t_masked_score.reserve(block_size);
				t_attension_prob.reserve(block_size);
				t_attension_out.reserve(block_size);
				t_merged_heads.reserve(block_size);
				t_attension_projected.reserve(block_size);
				t_attension_mask.reserve(block_size);
				t_attension_residual.reserve(block_size);

				t_gelu_input.reserve(block_size);
				t_gelu_output.reserve(block_size);
				t_mlp_mask.reserve(block_size);

				t_linear1_input.reserve(block_size);
				t_linear1_output.reserve(block_size);

				t_linear2_input.reserve(block_size);
				t_linear2_output.reserve(block_size);

				t_mlp_residual.reserve(block_size);

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
					
					t_l1_X.push_back(X_input2);
					
					vector<vector<float>> t2_l1_X_STD;
					vector<float> t2_l1_mean;
					vector<float> t2_l1_var;
					vector<float> t2_l1_std;
					
					t2_l1_X_STD.reserve(seq_len);
					t2_l1_mean.reserve(seq_len);
					t2_l1_var.reserve(seq_len);
					t2_l1_std.reserve(seq_len);

					Tensor::layer_norm(X_input2, gamma1[i], beta1[i], t2_l1_mean, t2_l1_var, t2_l1_std, t2_l1_X_STD);

					t_l1_mean.push_back(t2_l1_mean);
					t_l1_var.push_back(t2_l1_var);
					t_l1_std.push_back(t2_l1_std);
					t_l1_X_STD.push_back(t2_l1_X_STD);

					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Block Linear Projecting....." : cout << "";

					t_QKV_input.push_back(X_input2);
					
					t_Q_cache.push_back(q);
					t_K_cache.push_back(k);
					t_V_cache.push_back(v);
					
					t_Q_cache_H.push_back(q_h);
					t_K_cache_H.push_back(k_h);
					t_V_cache_H.push_back(v_h);
					
					linear_projection(X_input2, i);
					
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Block attension Score Calculating....." : cout << "";
					vector<vector<vector<float>>> t2_attension_score;
					vector<vector<vector<float>>> t2_scaled_score;
					vector<vector<vector<float>>> t2_masked_score;
					vector<vector<vector<float>>> t2_attension_prob;
					vector<vector<vector<float>>> t2_attension_out;
					vector<vector<float>> t2_merged_heads;
					
					t2_attension_score.reserve(head_size);
					t2_scaled_score.reserve(head_size);
					t2_masked_score.reserve(head_size);
					t2_attension_prob.reserve(head_size);
					t2_attension_out.reserve(head_size);
					t2_merged_heads.reserve(head_size);

					auto attension = Attension::score(q_h, k_h, v_h, wo[i], t2_attension_score, t2_scaled_score, 
														t2_masked_score, t2_attension_prob, t2_attension_out, t2_merged_heads);
					t_attension_score.push_back(t2_attension_score);
					t_scaled_score.push_back(t2_scaled_score);
					t_masked_score.push_back(t2_masked_score);
					t_attension_prob.push_back(t2_attension_prob);
					t_attension_out.push_back(t2_attension_out);
					t_merged_heads.push_back(t2_merged_heads);
					t_attension_projected.push_back(attension);

					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Attension Score Dropouting......" : cout << "";
					dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
					t_attension_mask.push_back(dropout_mask);
					X_input2 = Tensor::dropout(attension, dropout_mask, dropout_prob);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "First Residual Adding......" : cout << "";
					t_attension_residual.push_back(residual);
					X_input2 = Tensor::matadd(residual, X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					residual = X_input2;
					t_mlp_residual.push_back(residual);

					flag ? cout << "Block Second Layer normalizing....." : cout << "";
					t_l2_X.push_back(X_input2);
					
					vector<vector<float>> t2_l2_X_STD;
					vector<float> t2_l2_mean;
					vector<float> t2_l2_var;
					vector<float> t2_l2_std;
					
					t2_l2_X_STD.reserve(seq_len);
					t2_l2_mean.reserve(seq_len);
					t2_l2_var.reserve(seq_len);
					t2_l2_std.reserve(seq_len);

					Tensor::layer_norm(X_input2, gamma2[i], beta2[i], t2_l2_mean, t2_l2_var, t2_l2_std, t2_l2_X_STD);
					
					t_l2_mean.push_back(t2_l2_mean);
					t_l2_var.push_back(t2_l2_var);
					t_l2_std.push_back(t2_l2_std);
					t_l2_X_STD.push_back(t2_l2_X_STD);
					flag ? cout << "Done..." << endl << endl : cout << "";

					flag ? cout << "Linear Layer Started......" << endl : cout << "";
					
					flag ? cout << "Linear1 Calulating......" : cout << "";
					t_linear1_input.push_back(X_input2);
					X_input2 = Tensor::dot_product(X_input2, w1[i]);
					t_linear1_output.push_back(X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Gelu Calculating......" : cout << "";
					t_gelu_input.push_back(X_input2);
					Function::gelu(X_input2);
					t_gelu_output.push_back(X_input2);
					flag ? cout << "Done..." << endl : cout << "";

					flag ? cout << "Linear2 Calulating......" : cout << "";
					t_linear2_input.push_back(X_input2);
					X_input2 = Tensor::dot_product(X_input2, w2[i]);
					t_linear2_output.push_back(X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Linear Layer Calculated...." << endl << endl : cout << "";

					flag ? cout << "Linear Output Dropouting......" : cout << "";
					dropout_mask = Tensor::dropout_mask(seq_len, embed_size, dropout_rate);
					t_mlp_mask.push_back(dropout_mask);
					X_input2 = Tensor::dropout(attension, dropout_mask, dropout_prob);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "Second Residual Adding......" : cout << "";
					X_input2 = Tensor::matadd(residual, X_input2);
					flag ? cout << "Done..." << endl : cout << "";
					
					flag ? cout << "=========================================" << endl : cout << "";
					flag ? cout << i + 1 << " ended...." << endl : cout << "";
					flag ? cout << "=========================================" << endl << endl : cout << "";
				}
				
				l1_X.push_back(t_l1_X);
				l1_mean.push_back(t_l1_mean);
				l1_var.push_back(t_l1_var);
				l1_std.push_back(t_l1_std);
				l1_X_STD.push_back(t_l1_X_STD);

				l2_X.push_back(t_l2_X);
				l2_mean.push_back(t_l2_mean);
				l2_var.push_back(t_l2_var);
				l2_std.push_back(t_l2_std);
				l2_X_STD.push_back(t_l2_X_STD);

				QKV_input.push_back(t_QKV_input);
				Q_cache.push_back(t_Q_cache);
				K_cache.push_back(t_K_cache);
				V_cache.push_back(t_V_cache);

				Q_cache_H.push_back(t_Q_cache_H);
				K_cache_H.push_back(t_K_cache_H);
				V_cache_H.push_back(t_V_cache_H);

				attension_score.push_back(t_attension_score);
				scaled_score.push_back(t_scaled_score);
				masked_score.push_back(t_masked_score);
				attension_prob.push_back(t_attension_prob);
				attension_out.push_back(t_attension_out);
				merged_heads.push_back(t_merged_heads);
				attension_projected.push_back(t_attension_projected);
				attension_mask.push_back(t_attension_mask);
				attension_residual.push_back(t_attension_residual);

				mlp_residual.push_back(t_mlp_residual);
				
				gelu_input.push_back(t_gelu_input);
				gelu_output.push_back(t_gelu_output);
				mlp_mask.push_back(t_mlp_mask);

				linear1_input.push_back(t_linear1_input);
				linear1_output.push_back(t_linear1_output);

				linear2_input.push_back(t_linear2_input);
				linear2_output.push_back(t_linear2_output);

				flag ? cout << "Final Layer normalizing....." : cout << "";
	
				vector<vector<float>> t2_final_X_STD;
				vector<float> t2_final_mean;
				vector<float> t2_final_var;
				vector<float> t2_final_std;

				final_X.push_back(X_input2);
				Tensor::layer_norm(X_input2, final_gamma, final_beta, t2_final_mean, t2_final_var, t2_final_std, t2_final_X_STD);
				
				final_mean.push_back(t2_final_mean);
				final_var.push_back(t2_final_var);
				final_std.push_back(t2_final_std);
				final_X_STD.push_back(t2_final_X_STD);

				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "LM Head Projecting....." : cout << "";
				hidden_states.push_back(X_input2);
				X_input2 = Tensor::dot_product(X_input2, w_vocab);
				logits.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "Softmax....." : cout << "";
				Function::softmax(X_input2);
				softmax_probs.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";
				
				flag ? cout << "Calculating Loss....." : cout << "";
				for (int lss = 0; lss < seq_len; ++lss) loss += -log(X_input2[lss][Y[batch][lss]]);
				X_input.push_back(X_input2);
				flag ? cout << "Done..." << endl : cout << "";

				flag ? cout << "Calculating Gradients....." : cout << "";
				vector<vector<float>> gradient;
				vector<long long> t_target_ids;
				gradient.reserve(seq_len);
				t_target_ids.reserve(seq_len);
				for (int lss = 0; lss < seq_len; ++lss)
				{
					X_input2[lss][Y[batch][lss]] -= 1.0f;
					t_target_ids.push_back(Y[batch][lss]);
					gradient.push_back(X_input2[lss]);
				}
				d_logits.push_back(gradient);
				target_ids.push_back(t_target_ids);
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
			
			vector<vector<float>> dw_vocab(embed_size, vector<float>(vocab_size, 0.0f));
			vector<vector<vector<float>>> dh;
			
			auto embed_mat_t2 = Tensor::transpose(w_vocab);
			
			dh.reserve(batch_size);
			
			flag ? cout << " LM Head Backwarding...." << endl : cout << "";
			for (int gra = 0; gra < batch_size; ++gra)
			{
				auto h_t = Tensor::transpose(hidden_states[gra]);
				auto sum = Tensor::dot_product(h_t, d_logits[gra]);
				dw_vocab = Tensor::matadd(dw_vocab, sum);

				dh.push_back(Tensor::dot_product(d_logits[gra], embed_mat_t2));
			}
			flag ? cout << "Done..." << endl << endl : cout << "";

			flag ? cout << "Final Layer Norm Backward....." : cout << "";
			auto dg = Tensor::matmul_e(dh, final_X_STD);
			auto dfinal_gamma = Tensor::sum(dg);
			auto dfinal_beta = Tensor::sum(dh);
			auto dx_hat = Tensor::normalized_gradient(dh, final_gamma);
			auto dvar = Tensor::variance_gradient(final_X, dx_hat, final_mean, final_var);
			auto dmean = Tensor::mean_gradient(dx_hat, final_X_STD, dvar, final_std);
			auto dx = Tensor::input_gradient(dx_hat, final_X_STD, dvar, dmean, final_std);
			flag ? cout << "Done..." << endl: cout << "";
			
			vector<vector<vector<float>>> dw2(block_size, vector<vector<float>> (hidden_size, vector<float>(embed_size, 0.0f)));
			vector<vector<vector<float>>> dw1(block_size, vector<vector<float>> (embed_size, vector<float>(hidden_size, 0.0f)));
			vector<vector<float>> dgamma2(block_size, vector<float>(embed_size, 0.0f));
			vector<vector<float>> dbeta2(block_size, vector<float>(embed_size, 0.0f));

			flag ? cout << "Transformer Backward....." : cout << "";
			for (int back_batch = 0; back_batch < batch_size; ++back_batch)
			{
				auto d_mlp_residual = dx[back_batch];
				auto dmlp = dx[back_batch];
				for (int back_block = block_size - 1; back_block >= 0; back_block--)
				{
					dmlp = Tensor::elementwise_mul(dmlp, mlp_mask[back_batch][back_block]);
					
					auto gelu_output_t = Tensor::transpose(gelu_output[back_batch][back_block]);
					auto sum = Tensor::dot_product(gelu_output_t, dmlp);
					dw2[back_block] = Tensor::matadd(dw2[back_block], sum);

					auto t_w2 = Tensor::transpose(w2[back_block]);
					auto dgelu = Tensor::dot_product(dmlp, t_w2);

					auto gelu_d = gelu_input[back_batch][back_block];
					Function::gelu_derivative(gelu_d);
					auto dlinear1 = Tensor::elementwise_mul(dgelu, gelu_d);

					auto l2_X_STD_t = Tensor::transpose(l2_X_STD[back_batch][back_block]);
					sum = Tensor::dot_product(l2_X_STD_t, dlinear1);
					dw1[back_block] = Tensor::matadd(dw1[back_block], sum);

					auto t_w1 = Tensor::transpose(w1[back_block]);
					auto dln2 = Tensor::dot_product(dlinear1, t_w1);
					
					sum = Tensor::elementwise_mul(dln2, l2_X_STD[back_batch][back_block]);
					auto sum1 = Tensor::sum(sum);
					dgamma2[back_block] = Tensor::matadd(dgamma2[back_block],sum1);
					sum1 = Tensor::sum(dln2);
					dbeta2[back_block] = Tensor::matadd(dbeta2[back_block], sum1);

					auto dxhat = Tensor::elementwise_mul(dln2, gamma2[back_block]);
					
					break;
				}
				break;
			}
			break;
			flag ? cout << "Done..." << endl: cout << "";

			flag ? cout << "Batch " << ct << "Backward pass ended...." << endl : cout << "";
			flag ? cout << "================================================================" << endl << endl : cout << "";

			flag ? cout << "Batch " << ct << " ended...." << endl : cout << "";
		}
	}
};

#endif