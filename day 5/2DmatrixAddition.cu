#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixAdd(const float *A, const float *B, float *C, int cols, int rows) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	for (; row < rows; row += stride_y) {
		for (int j = col; j < cols; j += stride_x) {
			int idx =  row * cols + j;
			C[idx] = A[idx] + B[idx];
		}
	}
}

int main() {
	const int ROWS = 100;
	const int COLS = 100;
	size_t size = ROWS * COLS * sizeof(float);
	const dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32)};
	const int numTests = 2;

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	for (int i = 0; i < ROWS * COLS; i++) {
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
		dim3 blockSize = blockSizes[t];
		dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x,
									(ROWS + blockSize.y - 1) / blockSize.y);
		printf("\nTesting block size: %dx%d (grid: %d%d)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

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

		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		printf("first 5 results (row 0):\n");
		for (int j = 0; j < 5; j++) {
			int idx = 0 * COLS + j;
			printf("C[0][%d] = %.2f (A[0][%d] = %.2f + B[0][%d] = %.2f)\n", j, h_C[idx], j, h_A[idx], j, h_C[idx]);
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

	return 0;
}
