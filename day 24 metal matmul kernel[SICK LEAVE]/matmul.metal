#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiplication(
	device const float* A [[buffer(0)]],
	device const float* B [[buffer(1)]],
	device float* C [[buffer(2)]],

	const uint &M [[buffer(3)]],
	const uint &N [[buffer(4)]],
	const uint &K [[buffer(5)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= M || gid.y >= N) {
		return;
	}

	float sum = 0.0f;
	for (uint k = 0; k < K; k++) {
		sum += A[gid.x * K + k] * B[k * N + gid.y];
	}

	C[gid.x * N + gid.y] = sum;
}

