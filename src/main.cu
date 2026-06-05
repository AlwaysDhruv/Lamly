#include <iostream>
#include <string>
#include <vector>
#include "./include/BPE.hpp"
#include "./include/Transformer2.cuh"
using namespace std;

int main(int argc, char const *argv[])
{
	Tokenization tk;

	vector<string> tokens;
	vector<long long> token_ids;

	cout << "Encoding starts for test.txt....";

	tk.encoding("../data/test.txt", tokens, token_ids);

	cout << "done with " << token_ids.size() << " ids....." << endl;

	Transformer tr(token_ids);

	tr.input_embedding();
	
	tr.Transformers();

	return 0;
}