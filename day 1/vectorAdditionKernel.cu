#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (; idx < size; idx += blockDim.x * gridDim.x) {
		c[idx] = a[idx] + b[idx];
	}
}

int main() {
	const int N = 1024;
	size_t size = N * sizeof(float);

	float *h_a = (float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);

	for (int i = 0; i < N; i++) {
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaDeviceSynchronize();

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	printf("first 5 results:\n");
	for (int i = 0; i < 5; i++) {
		printf("c[%d] = %.2f (a[%d] = %.2f + b[%d] = %.2f)\n", i, h_c[i], i, h_a[i], i, h_b[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
