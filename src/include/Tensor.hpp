#ifndef TENSORS
#define TENSORS

#include <iostream>
#include <vector>
#include <tuple>
#include "Display.hpp"

using namespace std;

namespace Tensor
{
    vector<float> matadd(vector<float>& v1, vector<float>& v2)
    {
        int size = v1.size();
        
        vector<float> v3;
        v3.reserve(size);

        for (int i = 0; i < size; ++i) v3.push_back(v1[i] + v2[i]);
        return v3;
    }

    vector<vector<float>> matadd(vector<vector<float>>& v1, vector<vector<float>>& v2)
    {
        int dim1 = v1.size();
        int dim2 = v1[0].size();

        vector<vector<float>> ans;
        ans.reserve(dim1);

        for (int i = 0; i < dim1; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int k = 0; k < dim2; ++k) temp.push_back(v1[i][k] + v2[i][k]);
            ans.push_back(temp);
        }
        return ans;
    }

    vector<vector<float>> dot_product(
        const vector<vector<float>>& A,
        const vector<vector<float>>& B)
    {
        int m = A.size();
        int n = A[0].size();
        int p = B[0].size();

        vector<vector<float>> C(
            m,
            vector<float>(p, 0.0f));

        for (int i = 0; i < m; i++)
        {
            for (int k = 0; k < n; k++)
            {
                float temp = A[i][k];

                for (int j = 0; j < p; j++)
                {
                    C[i][j] += temp * B[k][j];
                }
            }
        }

        return C;
    }

    vector<vector<float>> dot_product(
        vector<vector<float>>& A,
        vector<vector<float>>& B)
    {
        int m = A.size();
        int n = A[0].size();
        int p = B[0].size();

        vector<vector<float>> C(
            m,
            vector<float>(p, 0.0f));

        for (int i = 0; i < m; i++)
        {
            for (int k = 0; k < n; k++)
            {
                float temp = A[i][k];

                for (int j = 0; j < p; j++)
                {
                    C[i][j] += temp * B[k][j];
                }
            }
        }

        return C;
    }

    void causal_mask(vector<vector<float>>& attension_score, float scale, vector<vector<float>>& scaled, vector<vector<float>>& masked)
    {
        for (size_t i = 0; i < attension_score.size(); ++i)
        {
            for (size_t k = 0; k < attension_score[0].size(); ++k)
            {
                if (k > i)
                {
                    attension_score[i][k] = -1e9f;
                    scaled[i][k] = -1e9f;
                }
                else
                {
                    attension_score[i][k] /= scale;
                    masked[i][k] /= scale;
                }
            }
        }
    }

    vector<vector<float>> elementwise_mul(vector<vector<float>>& v1, vector<vector<float>>& v2)
    {
        int dim1 = v1.size();
        int dim2 = v1[0].size();

        vector<vector<float>> ans;
        ans.reserve(dim1);

        for (int i = 0; i < dim1; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int j = 0; j < dim2; ++j) temp.push_back(v1[i][j] * v2[i][j]);
            ans.push_back(temp);
        }
        return ans;
    }
    
    vector<vector<float>> elementwise_mul(vector<vector<float>>& v1, vector<float>& v2)
    {
        int dim1 = v1.size();
        int dim2 = v1[0].size();

        vector<vector<float>> ans;
        ans.reserve(dim1);

        for (int i = 0; i < dim1; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int j = 0; j < dim2; ++j) temp.push_back(v1[i][j] * v2[j]);
            ans.push_back(temp);
        }
        return ans;
    }

    vector<vector<float>> dropout(vector<vector<float>>& v1, vector<vector<float>>& v2, float prob)
    {
        int dim1 = v1.size();
        int dim2 = v1[0].size();

        vector<vector<float>> ans;
        ans.reserve(dim1);

        for (int i = 0; i < dim1; ++i)
        {
            vector<float> temp1;
            temp1.reserve(dim2);
            for (int j = 0; j < dim2; ++j) temp1.push_back((v1[i][j] * v2[i][j]) / prob);
            ans.push_back(temp1);
        }
        return ans;
    }

    vector<vector<vector<float>>> head_spliting(vector<vector<float>>& v1, int head_size)
    {
        int seq_len = v1.size();
        int embed_size = v1[0].size();
        int head_dim = embed_size / head_size;

        vector<vector<vector<float>>> h;
        h.reserve(seq_len);

        for (int j = 0; j < seq_len; ++j)
        {
            vector<vector<float>> temp1;
            temp1.reserve(embed_size);      

            for (int k = 0; k < embed_size; k+=head_dim)
            {
                vector<float> temp;
                temp.reserve(head_dim);

                for (int l = k; l < k + head_dim; ++l) temp.push_back(v1[j][l]);
                temp1.push_back(temp);
            }
            h.push_back(temp1);
        }
        return h;
    }

    vector<vector<vector<float>>> split_heads(
        const vector<vector<float>>& x,
        int num_heads
    )
    {
        int seq_len = x.size();
        int embed_size = x[0].size();

        int head_dim = embed_size / num_heads;

        vector<vector<vector<float>>> out(
            num_heads,
            vector<vector<float>>(
                seq_len,
                vector<float>(head_dim)
            )
        );

        for (int t = 0; t < seq_len; t++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                for (int d = 0; d < head_dim; d++)
                {
                    out[h][t][d] =
                        x[t][h * head_dim + d];
                }
            }
        }

        return out;
    }

    vector<vector<float>> transpose(const vector<vector<float>>& v)
    {
        int dim1 = v.size();
        int dim2 = v[0].size();

        vector<vector<float>> trans;
        trans.reserve(dim1);

        for (int i = 0; i < dim2; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int j = 0; j < dim1; ++j) temp.push_back(v[j][i]);
            trans.push_back(temp);
        }
        
        return trans;
    }
    
    vector<vector<float>> transpose2(vector<vector<float>>& v)
    {
        int dim1 = v.size();
        int dim2 = v[0].size();

        vector<vector<float>> trans;
        trans.reserve(dim1);

        for (int i = 0; i < dim2; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int j = 0; j < dim1; ++j) temp.push_back(v[j][i]);
            trans.push_back(temp);
        }
        
        return trans;
    }

    vector<vector<vector<float>>> transpose(const vector<vector<vector<float>>>& v)
    {
        int dim1 = v.size();
        int dim2 = v[0].size();
        int dim3 = v[0][0].size();

        vector<vector<vector<float>>> trans;
        trans.reserve(dim2);

        for (int i = 0; i < dim2; ++i)
        {
            vector<vector<float>> temp;
            temp.reserve(dim1);

            for (int j = 0; j < dim1; ++j)
            {
                vector<float> temp1;
                temp1.reserve(dim3);

                for (int k = 0; k < dim3; ++k) temp1.push_back(v[j][i][k]);
                
                temp.push_back(temp1);
            }
            trans.push_back(temp);
        }        
        return trans;
    }

    void layer_norm(vector<vector<float>>& value, vector<float>& gamma, vector<float>& beta, vector<float>& m, vector<float>& v, vector<float>& s, vector<vector<float>>& X_norm)
    {
        int seq_len = value.size();
        int embed_size = value[0].size();
                
        m.reserve(seq_len);
        v.reserve(seq_len);
        s.reserve(seq_len);
        X_norm.reserve(seq_len);

        for (int j = 0; j < seq_len; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < embed_size; ++k) sum += value[j][k];
            float mean = sum / embed_size;
            m.push_back(mean);

            sum = 0.0;
            for (int k = 0; k < embed_size; ++k) sum += ((value[j][k] - mean) * (value[j][k] - mean));
            float variance = sum / embed_size;
            v.push_back(variance);
            
            vector<float> temp;
            temp.reserve(embed_size);
            float std = sqrt(variance + 1e-5f);
            s.push_back(1.0f / std);
            for (int k = 0; k < embed_size; ++k)
            {
                value[j][k] = (value[j][k] - mean) / std;
                temp.push_back(value[j][k]);
            }
            X_norm.push_back(temp);
            for (int k = 0; k < embed_size; ++k) value[j][k] = gamma[k] * value[j][k] + beta[k];
        }
    }

    vector<float> merge_head(vector<vector<float>>& v)
    {
        int cols = v.size();
        int rows = v[0].size();

        vector<float> ans;
        ans.reserve(cols * rows);
        for (int i = 0; i < cols; ++i) for (int j = 0; j < rows; ++j) ans.push_back(v[i][j]);
        return ans;
    }

    vector<vector<float>> merge_heads(
        const vector<vector<vector<float>>>& x
    )
    {
        int heads = x.size();
        int seq_len = x[0].size();
        int head_dim = x[0][0].size();

        int embed_size =
            heads * head_dim;

        vector<vector<float>> out(
            seq_len,
            vector<float>(
                embed_size,
                0.0f
            )
        );

        for (int token = 0;
             token < seq_len;
             ++token)
        {
            for (int head = 0;
                 head < heads;
                 ++head)
            {
                for (int dim = 0;
                     dim < head_dim;
                     ++dim)
                {
                    out[token]
                       [head * head_dim + dim]
                    =
                    x[head]
                     [token]
                     [dim];
                }
            }
        }

        return out;
    }
    
    vector<float> sum(vector<vector<vector<float>>>& v)
    {
        int batch_size = v.size();
        int seq_len = v[0].size();
        int embed_size = v[0][0].size();

        vector<float> ans;
        ans.reserve(embed_size);
        for (int j = 0; j < embed_size; ++j)
        {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; ++i) for (int k = 0; k < seq_len; ++k) sum += v[i][k][j];
            ans.push_back(sum);
        }
        return ans;
    }

    vector<float> sum(vector<vector<float>>& v)
    {
        int seq_len = v.size();
        int embed_size = v[0].size();

        vector<float> ans;
        ans.reserve(embed_size);
        for (int j = 0; j < embed_size; ++j)
        {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) sum += v[i][j];
            ans.push_back(sum);
        }
        return ans;
    }
    
    vector<vector<vector<float>>> matmul_e(vector<vector<vector<float>>>& v1, vector<vector<vector<float>>>& v2)
    {
        int batch_size = v1.size();
        int seq_len = v1[0].size();
        int embed_size = v1[0][0].size();

        vector<vector<vector<float>>> temp;
        temp.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i)
        {
            vector<vector<float>> temp1;
            temp1.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                vector<float> temp2;
                temp2.reserve(embed_size);
                for (int k = 0; k < embed_size; ++k) temp2.push_back(v1[i][j][k] * v2[i][j][k]);
                temp1.push_back(temp2);
            }
            temp.push_back(temp1);
        }

        return temp;
    }

    vector<vector<vector<float>>> normalized_gradient(vector<vector<vector<float>>>& v1, vector<float>& v2)
    {
        int batch_size = v1.size();
        int seq_len = v1[0].size();
        int embed_size = v1[0][0].size();
        vector<vector<vector<float>>> temp;
        temp.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i)
        {
            vector<vector<float>> temp1;
            temp1.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                vector<float> temp2;
                temp2.reserve(embed_size);
                for (int k = 0; k < embed_size; ++k) temp2.push_back(v1[i][j][k] * v2[k]);
                temp1.push_back(temp2);
            }
            temp.push_back(temp1);
        }

        return temp;                
    }

    vector<vector<float>> variance_gradient(vector<vector<vector<float>>>& dx_hat, vector<vector<vector<float>>>& X,
                                                    vector<vector<float>>& mean, vector<vector<float>>& var)

    {
        int batch_size = dx_hat.size();
        int seq_len = dx_hat[0].size();
        int embed_size = dx_hat[0][0].size();
        
        vector<vector<float>> ans;
        ans.reserve(batch_size);
        
        for (int i = 0; i < batch_size; ++i)
        {
            vector<float> temp;
            temp.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                float sum = 0.0f;
                float inv_std_cubed = powf(var[i][j] + 1e-5f, -1.5f);
                for (int k = 0; k < embed_size; ++k) sum += (dx_hat[i][j][k] * (X[i][j][k] - mean[i][j]) * (-0.5) * inv_std_cubed);
                temp.push_back(sum);
            }
            ans.push_back(temp);
        }
        return ans;
    }

    vector<vector<float>> mean_gradient(vector<vector<vector<float>>>& dx_hat, vector<vector<vector<float>>>& X_meaned,
                                        vector<vector<float>>& dvar, vector<vector<float>>& std)

    {
        int batch_size = dx_hat.size();
        int seq_len = dx_hat[0].size();
        int embed_size = dx_hat[0][0].size();
        
        vector<vector<float>> ans;
        ans.reserve(batch_size);
        
        for (int i = 0; i < batch_size; ++i)
        {
            vector<float> temp;
            temp.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                float value = -1.0f / std[i][j];
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                for (int k = 0; k < embed_size; ++k)
                {
                    sum1 += (dx_hat[i][j][k] * value);
                    sum2 += (-2.0f * X_meaned[i][j][k]);
                }
                float mean_term = (dvar[i][j] * sum2) / embed_size;
                temp.push_back(sum1 + mean_term);
            }
            ans.push_back(temp);
        }
        return ans;
    }

    vector<vector<vector<float>>> input_gradient(vector<vector<vector<float>>>& dx_hat, vector<vector<vector<float>>>& X_meaned,
                                                vector<vector<float>>& dvar, vector<vector<float>>& dmean, vector<vector<float>>& std)
    {
        int batch_size = dx_hat.size();
        int seq_len = dx_hat[0].size();
        int embed_size = dx_hat[0][0].size();
        
        vector<vector<vector<float>>> ans;
        ans.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i)
        {
            vector<vector<float>> temp1;
            temp1.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                vector<float> temp2;
                temp2.reserve(embed_size);
                for (int k = 0; k < embed_size; ++k) temp2.push_back((dx_hat[i][j][k] * std[i][j]) + (dvar[i][j] * 2.0f * X_meaned[i][j][k] / embed_size) + (dmean[i][j] / embed_size));
                temp1.push_back(temp2);
            }
            ans.push_back(temp1);
        }
        return ans;
    }
    
    void embed_pos_backward(vector<vector<float>>& dinput, vector<vector<float>>& dembed_mat, vector<vector<float>>& dpos_mat, vector<long long>& TX)
    {
        int seq_len = dinput.size();
        int embed_size = dinput[0].size();

        for (int tokens = 0; tokens < seq_len; ++tokens)
        {
            int token_id = TX[tokens];
            for (int em = 0; em < embed_size; ++em)
            {
                dembed_mat[token_id][em] += dinput[tokens][em];
                dpos_mat[tokens][em] += dinput[tokens][em];
            }
        }
    }

    void SGD(vector<float>& weigth, vector<float>& gradient, float lr)
    {
        int rows = weigth.size();

        for (int i = 0; i < rows; ++i)
        {
            weigth[i] -= lr * gradient[i];
        }
    }

    void SGD(vector<vector<float>>& weigth, vector<vector<float>>& gradient, float lr)
    {
        int rows = weigth.size();
        int cols = weigth[0].size();

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                weigth[i][j] -= lr * gradient[i][j];
            }
        }
    }

    void SGD(vector<vector<vector<float>>>& weigth, vector<vector<vector<float>>>& gradient, float lr)
    {
        int d1 = weigth.size();
        int d2 = weigth[0].size();
        int d3 = weigth[0][0].size();

        for (int i = 0; i < d1; ++i)
        {
            for (int j = 0; j < d2; ++j)
            {
                for (int k = 0; k < d3; ++k)
                {
                    weigth[i][j][k] -= lr * gradient[i][j][k];
                }
            }
        }
    }    
}
#endif