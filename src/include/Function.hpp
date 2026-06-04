#ifndef FUNCTION
#define FUNCTION

namespace Function
{
	float gelu_single(float x)
	{
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        const float coeff = 0.044715f;

        float x_cubed = std::pow(x, 3);
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        return 0.5f * x * (1.0f + tanh(inner));
    }

    float gelu_single_d(float x)
    {
        const float sqrt_2_over_pi =
            std::sqrt(2.0f / M_PI);

        const float coeff = 0.044715f;

        float x2 = x * x;
        float x3 = x2 * x;

        float u =
            sqrt_2_over_pi *
            (x + coeff * x3);

        float tanh_u = std::tanh(u);

        float sech2 =
            1.0f - (tanh_u * tanh_u);

        float du_dx =
            sqrt_2_over_pi *
            (1.0f + 3.0f * coeff * x2);

        return
            0.5f * (1.0f + tanh_u)
            +
            0.5f * x * sech2 * du_dx;
    }

    void gelu(vector<vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) H[i][j] = gelu_single(H[i][j]);
    }

    void gelu_derivative(vector<vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) H[i][j] = gelu_single_d(H[i][j]);
    }
    
    void softmax(vector<vector<float>>& matrix)
    {
        int rows = matrix.size();
        if (rows == 0) return;
        int cols = matrix[0].size();

        for (size_t i = 0; i < rows; ++i)
        {
            float max_val = *max_element(matrix[i].begin(), matrix[i].end());
            float sum = 0.0f;
            
            for (size_t j = 0; j < cols; ++j)
            {
                matrix[i][j] = exp(matrix[i][j] - max_val);
                sum += matrix[i][j];
            }

            for (size_t j = 0; j < cols; ++j)
            {
                if (sum > 0.0f) matrix[i][j] /= sum;
                else matrix[i][j] = 0.0f;
            }
        }
    }
}
#endif