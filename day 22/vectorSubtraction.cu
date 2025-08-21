#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorSub(const float* A, const float* B, float* C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		C[idx] = A[idx] - B[idx];
	}
}

void solve(const float* A, const float* B, float* C, int N) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	vectorSub<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: ", cudaGetErrorString(err));
	}
}

int main() {
	int N = 1 << 20;
	size_t bytes = N * sizeof(float);

	float* h_A = new float[N];
	float* h_B = new float[N];
	float* h_C = new float[N];

	for (int i = 0; i < N; i++) {
		h_A[i] = static_cast<float>(i + 1);
		h_B[i] = static_cast<float>(i * 0.5f);
	}

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc((void **)&d_C, bytes);

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	solve(d_A, d_B, d_C, N);

	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	printf("Sample results (first 5 elements):\n");
  for (int i = 0; i < 5 && i < N; i++) {
      printf("c[%d] = %f (a[%d] = %f, b[%d] = %f)\n", i, h_C[i], i, h_A[i], i, h_B[i]);
  }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}
