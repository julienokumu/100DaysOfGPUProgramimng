#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void matrixAdd(const float *A, const float *B, float *C, int cols, int rows) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	for (; rowIdx < rows; rowIdx += stride_y) {
		for (int j = colIdx; j < cols; j += stride_x) {
			int idx = rowIdx * cols + j;
			C[idx] = A[idx] + B[idx];
		}
	}
}

void matrixAddCPU(const float *A, const float *B, float *C, int cols, int rows) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			C[idx] = A[idx] + B[idx];
		}
	}
}

int main() {
	const int ROWS = 100;
	const int COLS = 100;
	size_t bytes = ROWS * COLS * sizeof(float);
	const dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32)};
	const int numTests = 2;

	float *h_A = (float *)malloc(bytes);
	float *h_B = (float *)malloc(bytes);
	float *h_C = (float *)malloc(bytes);
	float *h_C_cpu = (float *)malloc(bytes);

	for (int i = 0; i < ROWS * COLS; i++) {
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
		dim3 blockSize = blockSizes[t];
		dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x,
									(ROWS + blockSize.y - 1) / blockSize.y);
		printf("\nTesting block size: %dx%d (grid: %dx%d)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);

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

		cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

		auto cpu_start = std::chrono::high_resolution_clock::now();

		matrixAddCPU(h_A, h_B, h_C_cpu, ROWS, COLS);

		auto cpu_end = std::chrono::high_resolution_clock::now();

		float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
		printf("CPU execution time: %.3f ms\n", cpu_time);

		bool correct = true;
		for (int i = 0; i < ROWS * COLS; i++) {
			if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5) {
				int rowIdx = i / COLS, colIdx = i % COLS;
				printf("verification failed at [%d][%d]: GPU = %.2f, CPU = %.2f\n", rowIdx, colIdx, h_C[i], h_C_cpu[i]);

				correct = false;
				break;
			}
		}
		if (correct) {
			printf("matrix addition completed successfully!\n");
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
