#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixTranspose(const float* input, float* output, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && col < cols) {
		int input_idx = row * cols + col;
		int output_idx = col * rows + row;

		output[output_idx] = input[input_idx];
	}
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
											(cols + BLOCK_SIZE - 1) / BLOCKS_SIZE);

	matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
	}
}

int main() {
	int rows = 2;
	int cols = 3;
	
	float h_input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	float* h_output = (float*)malloc(cols * rows * sizeof(float));

	float *d_input, *d_output;
	cudaMalloc((void **)&d_input, rows * cols * sizeof(float));
	cudaMalloc((void **)&d_output, cols * rows * sizeof(float));

	cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
	
	solve(d_input, d_output, rows, cols);

	cudaMemcpy(h_output, d_output, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	return 0;
}
