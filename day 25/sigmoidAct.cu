#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void sigmoidAct(const float* x, float* y, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		y[idx] = 1.0f / (1.0f + expf(-x[idx]));
	}
}

void solve(const float* x, float* y, int N) {
	const int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	sigmoidAct<<<blocksPerGrid, threadsPerBlock>>>(x, y, N);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: ", cudaGetErrorString(err));
	}
}

int main() {
	int N = 1 << 20;
	size_t bytes = N * sizeof(float);

	float* h_x = new float[N];
	float* h_y = new float[N];

	for (int i = 0; i < N; i++) {
		h_x[i] = -0.5f + 10.0f * static_cast<float>(rand()) / RAND_MAX;
	}

	float *d_x, *d_y;
	cudaMalloc((void **)&d_x, bytes);
	cudaMalloc((void **)&d_y, bytes);

	cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

	solve(d_x, d_y, N);

	cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

	printf("First 5 sigmoid outputs: \n");
	for (int i = 0; i < 5 && i < N; i++) {
		printf("y[%d] = %f (x[%d] = %f)\n", i, h_y[i], i, h_x[i]);
	}

	cudaFree(d_x);
	cudaFree(d_y);
	free(h_x);
	free(h_y);

	return 0;
}

