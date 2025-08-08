#include <stdio.h>
#include <cuda_runtime.h>

__global__ void arrayMulOptimized(const float *A, const float *B, float *C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		C[idx] = A[idx] * B[idx];
	}
}

int main() {
	const int N = 10000;
	size_t bytes = N * sizeof(float);

	const int blockSizes[] = {64, 128, 256};
	const int numTests = 3;

	float *h_A = (float *)malloc(bytes);
	float *h_B = (float *)malloc(bytes);
	float *h_C = (float *)malloc(bytes);

	for (int i = 0; i < N; i++) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc((void **)&d_C, bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	for (int t = 0; t < numTests; t++) {
		int threadsPerBlock = blockSizes[t];
		int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

		printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start);

		arrayMulOptimized<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(err));
			
			return 1;
		}

		float gpu_time = 0;
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("GPU execution time: %.3f ms\n", gpu_time);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
