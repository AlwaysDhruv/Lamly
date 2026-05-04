#include <iostream>
#include <string>
#include <vector>
#include "./include/BPE.hpp"
#include "./include/Transformer.cuh"
using namespace std;

int main(int argc, char const *argv[])
{
	Tokenization tk;

	vector<string> tokens;
	vector<long long> token_ids;

	//tk.fit("../data/test.txt", 72);
	tk.encoding("../data/test2.txt", tokens, token_ids);

	Transformer tr(token_ids);

	tr.input_embedding();
	tr.linear_projection();

	return 0;
}