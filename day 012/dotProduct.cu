#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dotProduct(const float *A, const float *B, float *C, int N) {
	__shared__ float sharedMem[256];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float temp = 0.0f;
	if (idx < N) {
		temp = A[idx] * B[idx];
	}

	sharedMem[threadIdx.x] = temp;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
		if (threadIdx.x < stride) {
			sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(C, sharedMem[0]);
	}
}

void solve(const float *A, const float *B, float *C, int N) {
	const int threadsPerBlock = 256;
	int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

	cudaMemset(C, 0, sizeof(float));

	dotProduct<<<blocks, threadsPerBlock>>>(A, B, C, N);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
}

int main() {
	int N = 1000;
	size_t bytes = N * sizeof(float);

	float *h_A = new float[N];
	float *h_B = new float[N];
	float h_C;

	for (int i = 0; i < N; i++) {
		h_A[i] = static_cast<float>(rand()) / RAND_MAX;
		h_B[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	float *d_A, *d_B, *d_C;

	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc(&d_C, sizeof(float));

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
	
	solve(d_A, d_B, d_C, N);

	cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);

	return 0;
}
