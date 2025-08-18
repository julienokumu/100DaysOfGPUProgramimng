#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void GEMM(const float* A, const float* B, float* C, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n) {
		float sum = 0.0f;

		for (int l = 0; l < k; l++) {
			sum += A[row * k + l] * B[l * n + col];
		}

		C[row * n + col] = sum;
	}
}

void solve(const float* A, const float* B, float* C, int m, int n, int k) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

	GEMM<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, m, n, k);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
}

int main() {
	int m = 256;
	int n = 128;
	int k = 512;

	float* h_A = new float[m * k];
	float* h_B = new float[k * n];
	float* h_C = new float[m * n];

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			h_A[i * k + j] = static_cast<float>(i + j + 1);
		}
	}

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			h_B[i * n + j] = static_cast<float>(1.0 / (i + j + 1));
		}
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(float));
	cudaMalloc((void **)&d_B, k * n * sizeof(float));
	cudaMalloc((void **)&d_C, m * n * sizeof(float));

	cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

	solve(d_A, d_B, d_C, m, n, k);

	cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Sample results: \n");
	for (int i = 0; i < 5 && i < m * n; i++) {
		int row = i / n;
		int col = i % n;
		printf("C[d%][%d] = %f\n", row, col, h_C[i]);
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

