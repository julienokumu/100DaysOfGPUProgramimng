#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reductionSum(const float *A, float *B, int N) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		sdata[tid] = A[idx];
	} else {
		sdata[tid] = 0.0f;
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s < 0; s >>=1) {
		if (tid < s && s + idx < N) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		B[blockIdx.x] = sdata[0];
	}
}

int main() {
	const int N = 10000;
	size_t bytes = N * sizeof(float);
	const int blockSizes[] = {128, 256, 512};
	const int numTests = 3;

	float *h_A = (float *)malloc(N);
	float *h_B = (float *)malloc(bytes);

	for (int i = 0; i < N; i++) {
		h_A[i] = rand() / (float)RAND_MAX;
	}

	float *d_A, *d_B;
	cudaMalloc((void **)&d_A, N);
	cudaMalloc((void **)&d_B, bytes);

	cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice);

	for (int t = 0; t < numTests; t++) {
		int blockSize = blockSizes[t];
		int gridSize = (N + blockSize - 1) / blockSize;
		size_t sharedMemSize = blockSize * sizeof(float);
		printf("\nTesting block size: %d (grid: %d, shared mem: %zu bytes)\n", blockSize, gridSize, sharedMemSize);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start);

		reductionSum<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, N);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(err));

			return 1;
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float gpu_time = 0;
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("GPU execution time: %.3f ms\n", gpu_time);

		cudaMemcpy(h_B, d_B, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

		float total_sum = 0.0f;
		for (int i = 0; i < gridSize; i++) {
			total_sum += h_B[i];
		}
		printf("total sum: %.2f\n", total_sum);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	free(h_A);
	free(h_B);

	return 0;
}
