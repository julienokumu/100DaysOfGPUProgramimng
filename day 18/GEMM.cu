#include <stdio.h>
#include <cuda_runtime.h>

#define block_size 16

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
	dim3 threadsPerBlock(block_size, block_size);
	dim3 blocksPerGrid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);

	GEMM<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, m, n, k);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: s%\n", cudaGetErrorString(err));
	}
}

int main() {
	int m = 2;
	int k = 2;
	int n = 2;

	float* h_A = new float[m * k];
	float* h_B = new float[k * n];
	float* h_C = new float[m * n];

  h_A[0] = 1.0; h_A[1] = 2.0;
	h_A[2] = 3.0; h_A[3] = 2.0;

	h_B[0] = 5.0; h_B[1] = 6.0;
	h_B[2] = 7.0; h_B[3] = 8.0;

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(float));
	cudaMalloc((void **)&d_B, k * n * sizeof(float));
	cudaMalloc((void **)&d_C, m * n * sizeof(float));

	cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

	solve(d_A, d_B, d_C, m, n, k);

	cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nMatrix C: \n");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", h_C[i * n + j]);
		}
		printf("\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
