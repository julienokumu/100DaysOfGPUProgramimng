#include <metal.stdlib>
using namespace metal;

kernel void vectorAdd(
	device const float* A [[buffer(0)]],
	device const float* B [[buffer(1)]],
	device float* C [[buffer(2)]],
	uint idx [[thread_position_in_grid]]
) {
	C[idx] = A[idx] + B[idx];
}
