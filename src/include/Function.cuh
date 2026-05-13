#ifndef FUNCTION
#define FUNCTION

namespace Function
{
    __global__ void gelu_kernel(float *data, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
            const float coeff = 0.044715f;

            float x = data[i];

            float inner =
                sqrt_2_over_pi *
                (x + coeff * x * x * x);

            data[i] =
                0.5f * x * (1.0f + tanhf(inner));
        }
    }

    __global__ void softmax_kernel(
        float *data,
        int rows,
        int cols)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows)
        {
            int offset = row * cols;

            // max
            float max_val = data[offset];

            for (int j = 1; j < cols; j++)
            {
                if (data[offset + j] > max_val)
                    max_val = data[offset + j];
            }

            // exp + sum
            float sum = 0.0f;

            for (int j = 0; j < cols; j++)
            {
                data[offset + j] =
                    expf(data[offset + j] - max_val);

                sum += data[offset + j];
            }

            // normalize
            for (int j = 0; j < cols; j++)
            {
                data[offset + j] /= sum;
            }
        }
    }

    inline void gelu(std::vector<std::vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        int N = rows * cols;

        // flatten
        std::vector<float> flat(N);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = H[i][j];

        float *d_data;

        cudaMalloc(&d_data, N * sizeof(float));

        cudaMemcpy(d_data,
                   flat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        gelu_kernel<<<blocks, threads>>>(d_data, N);

        cudaMemcpy(flat.data(),
                   d_data,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_data);

        // reshape
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                H[i][j] = flat[i * cols + j];
    }

    inline void softmax(std::vector<std::vector<float>>& matrix)
    {
        int rows = matrix.size();
        int cols = matrix[0].size();

        int N = rows * cols;

        // flatten
        std::vector<float> flat(N);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = matrix[i][j];

        float *d_data;

        cudaMalloc(&d_data, N * sizeof(float));

        cudaMemcpy(d_data,
                   flat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (rows + threads - 1) / threads;

        softmax_kernel<<<blocks, threads>>>(
            d_data,
            rows,
            cols);

        cudaMemcpy(flat.data(),
                   d_data,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_data);

        // reshape
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = flat[i * cols + j];
    }    
}
#endif