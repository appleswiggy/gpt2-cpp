#pragma once
// ops.h - Low-level tensor operations for GPT-2 inference
// 
// All functions operate on raw float* buffers. Shapes are passed explicitly.
// Caller provides output buffers (No memory allocation happens here).
// 
// Convention: matrices are ROW major, shape is [rows, cols].

#include <cstddef>

namespace gpt2 {

// Matrix multiply
// out[M, N] = A[M, K] X B[K, N]
void matmul(float* out, const float* A, const float* B,
            int M, int K, int N);


// Layer Normalisation
// Normalises each row of x[rows, cols] independently.
// out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta
void layernorm(float* out, const float* x,
               const float* gamma, const float* beta,
               int rows, int cols, float eps = 1e-5f);


// Softmax
// In-place softmax over the last dimension: x[rows, cols]
// each row is independently softmaxed.
void softmax(float* x, int rows, int cols);


// GELU activation
// Approximate GELU (tanh version, same as GPT-2):
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// done in place on x[size].
void gelu(float* x, int size);

} // namespace gpt2
