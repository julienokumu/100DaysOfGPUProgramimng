#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU() {
	int threadID = threadId.x;
	printf("hello, GPU world! %d\n", threadID);
}

int main() {
	helloGPU<<<1, 32>>>();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaDeviceSynchornize();

	return 0;
}
