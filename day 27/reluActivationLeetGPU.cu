#include <cuda_runtime.h>
#include <math.h>

__global__ void reluActivation(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;
    size_t bytes = N * sizeof(float);

    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reluActivation<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
