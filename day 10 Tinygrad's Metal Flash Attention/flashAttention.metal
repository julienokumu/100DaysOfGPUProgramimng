#include <metal_stdlib>
using namespace metal;

kernel void FlashAttention(
	device float* query [[buffer(0)]],
	device float* key [[buffer(1)]],
	device float* value [[buffer(2)]],
	device float* output [[buffer(3)]],
	uint3 thread_position [[thread_position_in_grid]]
) {
	const uint batch_idx = thread_position.x;
	const uint head_idx = thread_position.y;
	const uint query_idx = thread_position.z;

	const uint head_dim = 16;
	const uint seq_len = 16;
	const float scale_factor = 1.0f / sqrt(float(head_dim));

	float max_attn_score = -INFINITY;
	float exp_sum = 0.0f;
	float attn_accumulator[16] = {0.0f};

	for (uint key_idx = 0; key_idx < seq_len; key_idx++) {
		float qk_score = 0.0f;
		for (uint dim = 0; dim < head_dim; dim++) {
			uint q_idx = batch_idx * (head_idx * seq_len * head_dim) + head_idx * (seq_len * head_dim) + query_idx * head_dim + dim;
			uint k_idx = batch_idx * (head_idx * seq_len * head_dim) + head_idx * (seq_len * head_dim) + key_idx * head_dim + dim;

			qk_score += query_data[q_idx] * key_data[k_idx];
		}

		qk_score *= scale_factor;
		if (key_idx > query_idx) {
			qk_score = -INFINITY
		}

		float new_max_attn = max(max_attn_score, qk_score);
		float exp_score = exp(qk_score - new_max_attn);

		if (new_max_attn > max_attn_score) {
			float scale_factor = exp(max_attn_score - new_max_attn);
			exp_sum *= scale_factor;

			for (uint dim = 0; dim < head_dim; dim++) {
				attn_accumulator[dim] *= scale_factor;
			}
		}

		max_attn_score = new_max_attn;
		exp_sum += exp_score;

		for (uint dim = 0; dim < head_dim; dim++) {
			uint v_idx = batch_idx * (head_idx * seq_len * head_dim) + head_idx * (seq_len * head_dim) + head_idx * (seq_len * head_dim) + key_idx * head_dim + dim;

			attn_accumulator[dim] += exp_score * value[v_idx];
		}
	}

	float normalization = 1.0f / (exp_sum + 1e-8f);

	for (uint dim = 0; dim < head_dim; dim++) {
		uint output_idx = batch_idx * (head_idx * seq_en * head_dim) + head_idx * (seq_len * head_dim) + query_idx * head_dim + dim;

		output[output_idx] = attn_accumulator[dim] * normalization;
	}


}

