// ops.cpp - Tensor operation implementations

#include "ops.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace gpt2 {

// Matrix multiplication
// out[M, N] = A[M, K] X B[K, N]
void matmul(float* out, const float* A, const float* B,
            int M, int K, int N) {
    // Zero the output first
    std::memset(out, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                out[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}


// Layer normalisation
// For each row (token embedding):
// 1. compute mean and variance
// 2. normalise: (x - mean) / sqrt(var + eps) [For each x in row]
// 3. Scale and shift using model params (gamma and beta) 
// note: model params are same for each row, but different for each column (token embedding dimensions)
// that is why the shape of model params is [n_embd]
void layernorm(float* out, const float* x,
               const float* gamma, const float* beta,
               int rows, int cols, float eps) {
    for (int i = 0; i < rows; i++) {
        const float* row = x + i * cols;
        float* out_row = out + i * cols;

        // mean and variance
        float mean = 0.0f;
        float var = 0.0f;

        for (int j = 0; j < cols; j++) {
            mean += row[j];
            float diff = row[j] - mean;
            var += diff * diff;
        }

        mean /= cols;
        var /= cols;

        // normalise, scale and shift
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < cols; j++) {
            out_row[j] = gamma[j] * ((row[j] - mean) * inv_std) + beta[j];
        }
    }
}


// Softmax - in place - applies to each row
// trick: we subtract max(x) from exponents before exponentiating:
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// 
// if you see the calculation, this value cancels out from the numerator and denominator
// it is only done to prevent overflow when logits are large positive numbers.
void softmax(float* x, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float* row = x + i * cols;

        // find max
        float max_val = row[0];
        for (int j = 0; j < cols; j++) {
            max_val = std::max(max_val, row[j]);
        }

        // exp(x - max) and accumulate sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        // normalise
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            row[j] *= inv_sum;
        }
    }
}


// GELU Activation - in place - applies to all elements of x
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// this is approximate (tanh) version of gelu, slightly different
// from the exact gelu = x * Φ(x)
// original GPT-2 also used tanh approximation
void gelu(float* x, int size) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/pi)

    for (int i = 0; i < size; i++) {
        float xi = x[i];
        float cube = xi * xi * xi;
        float inner = SQRT_2_OVER_PI * (xi + 0.044715f * cube);
        x[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
}

}