#ifndef WEIGTHS_H
#define WEIGTHS_H

#include <iostream>
#include <random>
#include <vector>

using namespace std;

namespace Random
{
	vector<vector<float>> random(int n1, int n2)
	{
		vector<vector<float>> weight;
        weight.reserve(n1);

        random_device rd;
	    mt19937 gen(rd());
	    
	    float std = 0.02f;
		normal_distribution<float> dist(0.0f, std);

	    for (int i = 0; i < n1; ++i)
	    {
	    	vector<float> temp;
	    	temp.reserve(n2);
	    	for (int j = 0; j < n2; ++j) temp.push_back(dist(gen));
	    	weight.push_back(temp);
	    }
        return weight;
	}

	vector<vector<vector<float>>> random(int n1, int n2, int n3)
	{
		vector<vector<vector<float>>> weight;
        weight.reserve(n1);

        random_device rd;
	    mt19937 gen(rd());
		
		float std = 0.02f;
		normal_distribution<float> dist(0.0f, std);

	    for (int i = 0; i < n1; ++i)
	    {
	    	vector<vector<float>> temp;
	    	temp.reserve(n2);
	    	for (int j = 0; j < n2; ++j)
	    	{
	    		vector<float> temp1;
	    		temp1.reserve(n3);
	    		for (int k = 0; k < n3; ++k) temp1.push_back(dist(gen));
	    		temp.push_back(temp1);
	    	}
	    	weight.push_back(temp);
	    }
        return weight;
	}

	vector<vector<float>> dropout_mask(int n1, int n2, float dropout_rate)
	{
		vector<vector<float>> mask;
		mask.reserve(n1);

        random_device rd;
	    mt19937 gen(rd());
	    uniform_real_distribution<float> dist(0.0f, 1.0f);

		for (int i = 0; i < n1; ++i)
		{
			vector<float> temp2;
			temp2.reserve(n2);
			
			for (int l = 0; l < n2; ++l)
			{
				float r = dist(gen);
			
				float mask = (r > dropout_rate) ? 1.0f : 0.0f;
            	temp2.push_back(mask);
			}
			mask.push_back(temp2);
		}
		return mask;
	}
};

#endif
