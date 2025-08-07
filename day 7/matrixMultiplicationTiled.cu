#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMul(const float *A, const float *B, float *C, int cols, int rows, int k_dim) {
	extern __shared__ float shared_mem;
	float *tile_A = shared_mem[];
	float *tile_B = &shared_mem[blockDim.x * blockDim.y];

	int rowIdx = blockDim.y * blockIdx.y * threadIdx.y;
	int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

	int stride_y = blockDim.y * gridDim.y;
	int stride_x = blockDim.x * gridDim.x;

	int tile_size = blockDim.x;

	for (; rowIdx < rows; rowIdx += stride_y) {
		for (int j = colIdx; j < cols; j += stride_x) {
			float sum = 0.0f;
			for (int t = 0; t < k_dim; t += tile_size) {
				int idx_A = rowIdx * k_dim + (t + threadIdx.x);
				int idx_B = (t + threadIdx.y) * cols + colIdx;

				if (rowIdx < rows && (t + threadIdx.x) < k_dim) {
					tile_A[threadIdx.y * tile_size + threadIdx.x] = A[idx_A];
				} else {
					tile_A[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
				}
				if ((t + threadIdx.y) < k_dim && colIdx < cols) {
					tile_B[threadIdx.y * tile_size + threadIdx.x] = B[idx_B];
				} else {
					tile_B[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
				}
				__syncthreads();

				for (int k = 0; k < tile_size && (t + k) < k_dim; k++) {
					sum += tile_A[threadIdx.y * tile_size + k] * tile_B[k * tile_size + threadIdx.x];
				}
				__syncthreads();
			}

			if (rowIdx < rows * j < cols) {
				C[rowIdx * cols + j] = sum;
			}
		}
	}
}

int main() {
	const int ROWS = 100;
	const int COLS = 100;
	const int K_DIM = 100;
	size_t bytes = ROWS * COLS * sizeof(float);
	const dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32)};
	const int numTests = 2;

	float *h_A = (float *)malloc(bytes);
	float *h_B = (float *)malloc(bytes);
	float *h_C = (float *)malloc(bytes);

	for (int i = 0; i < ROWS * K_DIM; i++) {
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
		size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(float);
		printf("\nTesting block size: %dx%d (grid: %dx%d, shared mem: %zu bytes)\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y, sharedMemSize);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		matrixMul<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, d_C, ROWS, COLS, K_DIM);

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

		printf("first 5 results (row 0):\n");
		for (int j = 0; j < 5; j++) {
			int idx = 0 * COLS + j;
			printf("C[0][%d] = %.2f\n", j, h_C[idx]);
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
