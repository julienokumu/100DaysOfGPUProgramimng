#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void FP16GEMM(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		float acc = 0.0f;
		for (int k = 0; k < K; k++) {
			acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
		}
		float c_val = __half2float(C[row * N + col]);
		float result = alpha * acc + beta * c_val;
		C[row * N + col] = __float2half(result);
	}
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	half *d_A, *d_B, *d_C;
	size_t bytes_A = M * K * sizeof(half);
	size_t bytes_B = K * N * sizeof(half);
	size_t bytes_C = M * N * sizeof(half);

	cudaMalloc((void **)&d_A, bytes_A);
	cudaMalloc((void **)&d_B, bytes_B);
	cudaMalloc((void **)&d_C, bytes_C);

	cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
											(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

	FP16GEMM<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: ", cudaGetErrorString(err));
	}

	cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
