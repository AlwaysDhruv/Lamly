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
	vector<long long> token_ids;

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
			token_ids = ids;
			context_len = token_ids.size();
			xy_size = context_len - 1;
			num_seq = xy_size - seq_len;
			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}

	void ready()
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

		auto embed_mat = Tensor::random(vocab_size, embed_size);
		auto pos_mat = Tensor::random(xy_size ,embed_size);

		vector<vector<float>> embed_x;
		embed_x.reserve(xy_size);
		
		for (int i = 0; i < xy_size; ++i) embed_x.push_back(Tensor::add(embed_mat[token_x[i]], pos_mat[i]));

		vector<vector<vector<float>>> X;
		X.reserve(num_seq);

		for (int i = 0; i < num_seq; ++i)
		{
			vector<vector<float>> temp;
			temp.reserve(seq_len + i);
			for (int j = 0 + i; j < seq_len + i; ++j)
			{
				temp.push_back(embed_x[j]);
			}
			X.push_back(temp);
		}
		Debug::display(X);
	}
};

#endif