#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void L2NormKernel(const float* input, float* output, float* globalSum, int N) {
	__shared__ float sharedMem[256];
	
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float acc = 0.0f;
	if (idx < N) {
		acc = input[idx] * input[idx];
	}
	sharedMem[tid] = acc;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
		if (tid < stride) {
			sharedMem[tid] += sharedMem[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(globalSum, sharedMem[0]);
	}
	__syncthreads();

	volatile float *norm = globalSum;
	float L2Norm = sqrtf(*norm);

	if (idx < N && L2Norm > 0.0f) {
			output[idx] = input[idx] / L2Norm;
	}
}

	void solve(const float* input, float* output, int N) {
		const int threadsPerBlock = 256;
		int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	
		float* d_globalSum;
		cudaMemset(d_globalSum, 0, sizeof(float));
		cudaMalloc((void **)&d_globalSum, sizeof(float));

		L2NormKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_globalSum, N);
		cudaDeviceSynchronize();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error: ", cudaGetErrorString(err));
		}

		cudaFree(d_globalSum);
	}

	int main() {
		int N = 1 << 20;
		size_t bytes = N * sizeof(float);

		float* h_input = new float[N];
		float* h_output = new float[N];

		for (int i = 0; i < N; i++) {
			h_input[i] = static_cast<float>(rand()) / RAND_MAX;
		}

		float *d_input, *d_output;
		cudaMalloc((void **)&d_input, bytes);
		cudaMalloc((void **)&d_output, bytes);

		cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

		solve(d_input, d_output, N);

		cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

		printf("First 5 normalized values:\n");
		for (int i = 0; i < 5 && i < N; i++) {
			printf("output[%d] = %f (input[%d] = %f)\n", i, h_output[i], i, h_input[i]);
		}

		cudaFree(d_input);
		cudaFree(d_output);
		free(h_input);
		free(h_output);

		return 0;
	}
 
