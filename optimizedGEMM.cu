#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void optimizedGEMM(const float* A, const float* B, float* C, int M, int N, int K) {
	__shared__ float tileA[TILE_DIM][TILE_DIM];
	__shared__ float tileB[TILE_DIM][TILE_DIM];

	int row = blockIdx.y * TILE_DIM + threadIdx.y;
	int col = blockIdx.x * TILE_DIM + threadIdx.x;

	float sum = 0.0f;
	for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++){
		int rowA = row;
		int colA = t * TILE_DIM + threadIdx.x;
		if (rowA < M && colA < K) {
			tileA[threadIdx.y][threadIdx.x] = A[rowA * K + colA];
		} else {
			tileA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		int rowB = t * TILE_DIM + threadIdx.y;
		int colB = col;
		if (rowB < K && colB < N) {
			tileB[threadIdx.y][threadIdx.x] = B[rowB * N + colB];
		} else {
			tileB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int l = 0; l < TILE_DIM; l++) {
			sum += tileA[threadIdx.y][l] * tileB[l][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
}

void tiledGEMM(const float* A, const float* B, float* C, int M, int N, int K) {
	dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
	dim3 blocksPerGrid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

	optimizedGEMM<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: ", cudaGetErrorString(err));
	}
}

int main() {
	int M = 256;
	int K = 512;
	int N = 128;

	float *h_A = new float[M * K];
	float *h_B = new float[K * N];
	float *h_C = new float[M * N];

	for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
          h_A[i * K + j] = static_cast<float>(i + j + 1);
      }
  }
  for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
          h_B[i * N + j] = static_cast<float>(1.0 / (i + j + 1));
      }
  }

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, M * K * sizeof(float));
	cudaMalloc((void **)&d_B, K * N * sizeof(float));
	cudaMalloc((void **)&d_C, M * N * sizeof(float));

	cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

	tiledGEMM(d_A, d_B, d_C, M, N, K);

	cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Sample results (first 5 elements of C):\n");
  for (int i = 0; i < 5 && i < M * N; i++) {
      int row = i / N;
      int col = i % N;
      printf("C[%d][%d] = %f\n", row, col, h_C[i]);
  }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
