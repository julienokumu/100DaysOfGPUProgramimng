#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

__global__ void maxReduction(const float* input, float* output, int N) {
	extern __shared__ float sharedMem[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	sharedMem[tid] = (idx < N) ? input[idx] : -FLT_MAX;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>=1) {
		if (tid < s) {
			sharedMem[tid] = fmaxf(sharedMem[tid], sharedMem[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		*output = sharedMem[0];
	}
}

int main() {
	const int N = 1024;
	size_t bytes = N * sizeof(float);
	srand(42);

	float* h_input = (float *)malloc(bytes);
	for (int i = 0; i < N; i++) {
		h_input[i] = (rand() / (float)RAND_MAX) * 100.0f;
	}

	float* d_input, *d_output;
	cudaMalloc((void **)&d_input, bytes);
	cudaMalloc((void **)*&d_output, bytes);

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	float h_output = -FLT_MAX;
	cudaMemcpy(d_output, &h_output, bytes, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t sharedMemSize = threadsPerBlock * sizeof(float);

	maxReduction<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
	cudaDeviceSynchronize();

	cudaMemcpy(&h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	printf("Max value: %f\n", h_output);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	return 0;
}
