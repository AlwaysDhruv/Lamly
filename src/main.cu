#include <iostream>
#include <string>
#include <vector>
#include "./include/BPE.hpp"
#include "./include/Transformer.cuh"

using namespace std;

void save_tokens(
    const vector<long long>& token_ids,
    const string& path)
{
    ofstream file(path, ios::binary);

    size_t size = token_ids.size();

    file.write(
        reinterpret_cast<const char*>(&size),
        sizeof(size));

    file.write(
        reinterpret_cast<const char*>(token_ids.data()),
        size * sizeof(long long));
}

vector<long long> load_tokens(
    const string& path)
{
    ifstream file(path, ios::binary);

    size_t size;

    file.read(
        reinterpret_cast<char*>(&size),
        sizeof(size));

    vector<long long> token_ids(size);

    file.read(
        reinterpret_cast<char*>(token_ids.data()),
        size * sizeof(long long));

    return token_ids;
}

int main(int argc, char const *argv[])
{   
    cout << "Token Ids Imprting from tokens.txt....";
    vector<long long> token_ids = load_tokens("../data/tokens.bin");
    cout << "done with " << token_ids.size() << " ids....." << endl;

    // //tk.fit("../data/test.txt", 1000);
    // cout << "Encoding starts for test.txt....";
    
    // tk.encoding("../data/test.txt", tokens, token_ids);
    // save_tokens(token_ids, "tokens.bin");
    // cout << "done with " << token_ids.size() << " ids....." << endl;

    Transformer tr(token_ids);

    tr.input_embedding();

    tr.Transformers();
    
    return 0;
}
