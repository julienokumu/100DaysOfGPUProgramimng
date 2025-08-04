#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void arrayMul(const float *A, const float *B, float *C, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (; idx < size; idx += blockDim.x * gridDim.x) {
		C[idx] = A[idx] * B[idx];
	}
}

void arrayMulCPU(const float *A, const float *B, float *C, int size) {
	for (int i = 0; i < size; i++) {
		C[i] = A[i] * B[i];
	}
}

int main() {
	const int N = 10000;
	size_t size = N * sizeof(float);
	const int blockSizes[] = {64, 128, 256};
	const int numTests = 3;

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);
	float *h_C_cpu = (float *)malloc(size);

	for (int i = 0; i < N; i++) {
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	for (int t = 0; t < numTests; t++) {
		int threadsPerBlock = blockSizes[t];
		int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
		
		printf("\nTesting block size: %d (blocks: %d)\n", threadsPerBlock, blocks);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		arrayMul<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(err));

			return 1;
		}

		cudaEventRecord(stop);
		cudaDeviceSynchronize(stop);

		float gpu_time = 0;
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("GPU execution time: %.3f ms\n", gpu_time);

		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		auto cpu_start = std::chrono::high_resolution_clock::now();
		arrayMulCPU(h_A, h_B, h_C_cpu, N);
		auto cpu_end = std::chrono::high_resolution_clock::now();

		float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
		printf("CPU execution time : %.3f ms\n", cpu_time);

		bool correct = true;
		for (int i = 0; i < N; i++) {
			if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5) {
				printf("verification failed at index %d: GPU = %.2f, CPU = %.2f\n", i, h_C[i], h_C_cpu[i]);

				correct = false;
				break;
			}
		}

		if (correct) {
			printf("multiplication completed successfully\n");
			printf("speedup: %.2fX\n", cpu_time / gpu_time);
		}

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_cpu);

	return 0;
}
