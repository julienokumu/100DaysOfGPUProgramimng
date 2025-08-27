#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixTranspose(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;

        output[output_idx] = input[input_idx];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    size_t bytesInput = rows * cols * sizeof(float);
    size_t bytesOutput = cols * rows * sizeof(float);

    cudaMalloc((void **)&d_input, bytesInput);
    cudaMalloc((void **)&d_output, bytesOutput);

    cudaMemcpy(d_input, input, bytesInput, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, bytesOutput, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
