#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>


template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t hidden_dim,
                              float eps) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t stride = (hidden_dim + blockDim.x - 1) / blockDim.x; // 每个线程处理的元素数

    // 1. 每个线程累加自己负责的多个元素的平方和
    float sum_sq = 0.0f;
    for (size_t s = 0; s < stride; ++s) {
        size_t col = tid * stride + s;
        if (col < hidden_dim) {
            float val = static_cast<float>(input[row * hidden_dim + col]);
            sum_sq += val * val;
        }
    }

    // 2. 共享内存归约（每个线程贡献一个 partial sum）
    __shared__ float s_partial[256];
    s_partial[tid] = sum_sq;
    __syncthreads();

    // 树形归约（要求 blockDim.x 是2的幂，这里满足）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算逆均方根并广播
    float inv_rms = 1.0f;
    if (tid == 0) {
        float mean = s_partial[0] / static_cast<float>(hidden_dim);
        inv_rms = rsqrtf(mean + eps);
    }
    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = inv_rms;
    __syncthreads();
    inv_rms = s_inv_rms;

    // 4. 归一化、加权并写回（每个线程处理相同范围的元素）
    for (size_t s = 0; s < stride; ++s) {
        size_t col = tid * stride + s;
        if (col < hidden_dim) {
            float val = static_cast<float>(input[row * hidden_dim + col]);
            float w   = static_cast<float>(weight[col]);
            output[row * hidden_dim + col] = static_cast<T>(val * inv_rms * w);
        }
    }
}


template <typename T>
__device__ __forceinline__ float attentionToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float attentionToFloat<half>(half value)
{
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T attentionFromFloat(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half attentionFromFloat<half>(float value)
{
    return __float2half_rn(value);
}


/*
 * 测试中的 head_dim 不超过 256。
 *
 * 每个 CUDA 线程负责一个完整的：
 *   (batch, query position, query head)
 *
 * 使用线程私有数组避免 block 归约改变 Q·K 的浮点加法顺序。
 */
constexpr int kAttentionMaxHeadDim = 256;


/*
 * 严格按照 dimension=0,1,... 的顺序执行 FP32 FMA。
 *
 * 不要改成：
 *
 *     dot += q * k;
 *
 * 也不要改成 block/warp reduction，否则 float 结果会因为
 * 浮点加法顺序发生变化。
 */
template <typename T>
__device__ __forceinline__ float attentionSerialDot(
    const float* __restrict__ query_cache,
    const T* __restrict__ key_row,
    int head_dim)
{
    float dot_product = 0.0f;

#pragma unroll 1
    for (int dimension = 0;
         dimension < head_dim;
         ++dimension) {
        const float key_value =
            attentionToFloat(key_row[dimension]);

        dot_product = fmaf(
            query_cache[dimension],
            key_value,
            dot_product);
    }

    return dot_product;
}


template <typename T>
__global__ void flashAttnKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal)
{
    const int linear_index =
        static_cast<int>(blockIdx.x) *
            static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);

    const int total_queries =
        batch_size * target_seq_len * query_heads;

    if (linear_index >= total_queries) {
        return;
    }

    /*
     * linear_index 的布局对应：
     *
     * [batch, target_seq_len, query_heads]
     */
    const int query_head =
        linear_index % query_heads;

    const int query_index =
        (linear_index / query_heads) % target_seq_len;

    const int batch_index =
        linear_index /
        (target_seq_len * query_heads);

    /*
     * GQA/MQA 映射。
     *
     * query_heads=8、kv_heads=2：
     *
     * Q heads 0,1,2,3 -> KV head 0
     * Q heads 4,5,6,7 -> KV head 1
     */
    const int query_heads_per_kv_head =
        query_heads / kv_heads;

    const int kv_head =
        query_head / query_heads_per_kv_head;

    const size_t query_offset =
        static_cast<size_t>(linear_index) *
        static_cast<size_t>(head_dim);

    float query_cache[kAttentionMaxHeadDim];
    float output_accumulator[kAttentionMaxHeadDim];

#pragma unroll 1
    for (int dimension = 0;
         dimension < head_dim;
         ++dimension) {
        query_cache[dimension] =
            attentionToFloat(
                query[query_offset + dimension]);

        output_accumulator[dimension] = 0.0f;
    }

    /*
     * 必须以 float 在 GPU 上计算。
     *
     * 不要在 host 端使用 double sqrt 后再传入，
     * 这会改变严格 float 测试中的舍入路径。
     */
    const float scale =
        1.0f /
        sqrtf(static_cast<float>(head_dim));

    /*
     * torch SDPA 的 causal mask 为左上对齐：
     *
     * Q0 -> K0
     * Q1 -> K0, K1
     * Q2 -> K0, K1, K2
     */
    int last_key_index = src_seq_len - 1;

    if (is_causal &&
        query_index < last_key_index) {
        last_key_index = query_index;
    }

    /*
     * 第一遍：求最大 attention score。
     */
    float maximum_score = -1.0f / 0.0f;

#pragma unroll 1
    for (int key_index = 0;
         key_index <= last_key_index;
         ++key_index) {
        const size_t key_offset =
            ((static_cast<size_t>(batch_index) *
                  static_cast<size_t>(src_seq_len) +
              static_cast<size_t>(key_index)) *
                 static_cast<size_t>(kv_heads) +
             static_cast<size_t>(kv_head)) *
            static_cast<size_t>(head_dim);

        const float dot_product =
            attentionSerialDot(
                query_cache,
                key + key_offset,
                head_dim);

        const float score =
            dot_product * scale;

        maximum_score =
            fmaxf(maximum_score, score);
    }

    /*
     * 第二遍：计算稳定 Softmax 分母。
     */
    float denominator = 0.0f;

#pragma unroll 1
    for (int key_index = 0;
         key_index <= last_key_index;
         ++key_index) {
        const size_t key_offset =
            ((static_cast<size_t>(batch_index) *
                  static_cast<size_t>(src_seq_len) +
              static_cast<size_t>(key_index)) *
                 static_cast<size_t>(kv_heads) +
             static_cast<size_t>(kv_head)) *
            static_cast<size_t>(head_dim);

        const float dot_product =
            attentionSerialDot(
                query_cache,
                key + key_offset,
                head_dim);

        const float score =
            dot_product * scale;

        const float unnormalized_probability =
            expf(score - maximum_score);

        denominator += unnormalized_probability;
    }

    /*
     * 只进行一次除法。第三遍使用乘法：
     *
     * probability = exp(score - max) * inverse_denominator
     */
    const float inverse_denominator =
        denominator > 0.0f
            ? 1.0f / denominator
            : 0.0f;

    /*
     * 第三遍：重新计算概率并完成 P·V。
     */
#pragma unroll 1
    for (int key_index = 0;
         key_index <= last_key_index;
         ++key_index) {
        const size_t kv_offset =
            ((static_cast<size_t>(batch_index) *
                  static_cast<size_t>(src_seq_len) +
              static_cast<size_t>(key_index)) *
                 static_cast<size_t>(kv_heads) +
             static_cast<size_t>(kv_head)) *
            static_cast<size_t>(head_dim);

        const float dot_product =
            attentionSerialDot(
                query_cache,
                key + kv_offset,
                head_dim);

        const float score =
            dot_product * scale;

        const float unnormalized_probability =
            expf(score - maximum_score);

        const float probability =
            inverse_denominator *
            unnormalized_probability;

#pragma unroll 1
        for (int dimension = 0;
             dimension < head_dim;
             ++dimension) {
            const float value_element =
                attentionToFloat(
                    value[kv_offset + dimension]);

            /*
             * 保持 probability * V + accumulator
             * 为一次舍入的 FP32 FMA。
             */
            output_accumulator[dimension] =
                fmaf(
                    probability,
                    value_element,
                    output_accumulator[dimension]);
        }
    }

#pragma unroll 1
    for (int dimension = 0;
         dimension < head_dim;
         ++dimension) {
        output[query_offset + dimension] =
            attentionFromFloat<T>(
                output_accumulator[dimension]);
    }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function
  T *input,*weight,*output;

  cudaMallocManaged(&input,h_input.size()*sizeof(T));
  cudaMallocManaged(&weight,h_weight.size()*sizeof(T));
  cudaMallocManaged(&output,h_output.size()*sizeof(T));

  cudaMemcpy(input, h_input.data(), h_input.size()*sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(weight, h_weight.data(), h_weight.size()*sizeof(T), cudaMemcpyHostToDevice);
 
  int device=-1;
  cudaGetDevice(&device);

  cudaMemAdvise(input,h_input.size()*sizeof(T),cudaMemAdviseSetReadMostly,device);
  cudaMemAdvise(weight,h_weight.size()*sizeof(T),cudaMemAdviseSetReadMostly,device);
  cudaMemAdvise(output,h_output.size()*sizeof(T),cudaMemAdviseSetPreferredLocation,device);

  cudaStream_t stream = 0; 
  cudaMemPrefetchAsync(input,h_input.size()*sizeof(T),device,stream);
  cudaMemPrefetchAsync(weight,h_weight.size()*sizeof(T),device,stream);


  size_t shared_mem = 256 * sizeof(T);
  rmsNormKernel<T><<<rows, 256, shared_mem,stream>>>(
        input, weight, output, hidden_dim, eps
    );

  cudaMemcpy(h_output.data(), output, h_output.size()*sizeof(T), cudaMemcpyDeviceToHost);
  
  cudaFree(input);
  cudaFree(weight);
  cudaFree(output);
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    // TODO: Implement the flash attention function
    constexpr int kMaxHeadDim = 256;
    constexpr int threads_per_block = 256;

    if (batch_size <= 0 ||
        target_seq_len <= 0 ||
        query_heads <= 0 ||
        kv_heads <= 0 ||
        head_dim <= 0) {
        return;
    }

    if (query_heads % kv_heads != 0) {
        return;
    }

    /*
     * 当前 kernel 的线程私有缓存支持 head_dim <= 256。
     */
    if (head_dim > kAttentionMaxHeadDim) {
        return;
    }

    const size_t query_element_count =
        static_cast<size_t>(batch_size) *
        static_cast<size_t>(target_seq_len) *
        static_cast<size_t>(query_heads) *
        static_cast<size_t>(head_dim);

    const size_t kv_element_count =
        static_cast<size_t>(batch_size) *
        static_cast<size_t>(src_seq_len) *
        static_cast<size_t>(kv_heads) *
        static_cast<size_t>(head_dim);

    if (h_q.size() < query_element_count ||
        h_k.size() < kv_element_count ||
        h_v.size() < kv_element_count ||
        h_o.size() < query_element_count) {
        return;
    }

    /*
     * 空 K/V 的安全处理。
     */
    if (src_seq_len <= 0) {
        for (size_t index = 0;
             index < query_element_count;
             ++index) {
            h_o[index] = T{};
        }

        return;
    }

    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_output = nullptr;

    cudaMalloc(
        reinterpret_cast<void**>(&d_query),
        query_element_count * sizeof(T));

    cudaMalloc(
        reinterpret_cast<void**>(&d_key),
        kv_element_count * sizeof(T));

    cudaMalloc(
        reinterpret_cast<void**>(&d_value),
        kv_element_count * sizeof(T));

    cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        query_element_count * sizeof(T));

    cudaMemcpy(
        d_query,
        h_q.data(),
        query_element_count * sizeof(T),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_key,
        h_k.data(),
        kv_element_count * sizeof(T),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_value,
        h_v.data(),
        kv_element_count * sizeof(T),
        cudaMemcpyHostToDevice);


    const size_t total_queries =
        static_cast<size_t>(batch_size) *
        static_cast<size_t>(target_seq_len) *
        static_cast<size_t>(query_heads);

    const unsigned int block_count =
        static_cast<unsigned int>(
            (total_queries + threads_per_block - 1) /
            threads_per_block);

    flashAttnKernel<T>
        <<<block_count, threads_per_block>>>(
            d_query,
            d_key,
            d_value,
            d_output,
            batch_size,
            target_seq_len,
            src_seq_len,
            query_heads,
            kv_heads,
            head_dim,
            is_causal);

    cudaMemcpy(
        h_o.data(),
        d_output,
        query_element_count * sizeof(T),
        cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
   
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);


