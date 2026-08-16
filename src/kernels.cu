#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
#include <algorithm>
#include <cmath>


template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t rows,
                              size_t hidden_dim,
                              float eps) {
    size_t row=blockIdx.x;
    size_t tid=threadIdx.x;
    size_t stride=blockDim.x;

    float sum=0.0f;
    for(size_t i=tid;i<hidden_dim;i+=stride){
         float val=input[row*hidden_dim+i];
        sum+=val*val;
    }

    extern __shared__ float smem[];
    smem[tid]=sum; 
    __syncthreads();

    for(size_t s=stride/2;s>0;s>>=1){
        if(tid<s){
            smem[tid]+=smem[tid+s];
        }
        __syncthreads();
    }

    float total_sum=smem[0];
    float rms=rsqrtf(total_sum / static_cast<float>(hidden_dim) + eps);

    for(size_t i=tid;i<hidden_dim;i+=stride){
        output[row*hidden_dim+i]=static_cast<T>(
        (static_cast<float>(input[row*hidden_dim+i]))
        *rms
        *static_cast<float>(weight[i])
        );
    }

}


template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* out,
                                       int batch_size, int target_seq_len, int src_seq_len,
                                       int query_heads, int kv_heads, int head_dim,
                                       bool is_causal, float scale) {
    // 每个线程处理一个 query 位置 (b, i, h)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_queries = batch_size * target_seq_len * query_heads;
    if (idx >= total_queries) return;

    int h = idx % query_heads;
    int tmp = idx / query_heads;
    int i = tmp % target_seq_len;
    int b = tmp / target_seq_len;

    int group_size = query_heads / kv_heads;
    int hk = h / group_size;

    // 查询向量缓存
    constexpr int kMaxHeadDim = 256;
    float q_cache[kMaxHeadDim];
    size_t q_offset = (size_t)idx * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        q_cache[d] = static_cast<float>(q[q_offset + d]);
    }

    // 全局在线 softmax 状态
    float m = -1e30f;
    float l = 0.0f;
    float acc[kMaxHeadDim];  // 每个维度一个累加器
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    // 分段参数
    constexpr int kSegmentSize = 32;  // 每段包含的 key 数量

    int last_key = src_seq_len - 1;
    if (is_causal && i < last_key) last_key = i;

    int j = 0;
    while (j <= last_key) {
        int seg_end = min(j + kSegmentSize - 1, last_key);
        // 段内局部最大值
        float local_m = -1e30f;
        // 存储段内每个 key 的 score（或即时计算）
        // 我们需先计算段内所有 key 的 score 得到 local_m
        for (int jj = j; jj <= seg_end; ++jj) {
            size_t k_offset = ((size_t)b * src_seq_len + jj) * kv_heads * head_dim + hk * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float k_val = static_cast<float>(k[k_offset + d]);
                score += q_cache[d] * k_val;  // 或使用 FMA
            }
            score *= scale;
            local_m = fmaxf(local_m, score);
        }

        // 计算段内 exp 和 weighted sum
        float local_l = 0.0f;
        float local_acc[kMaxHeadDim];
        for (int d = 0; d < head_dim; ++d) local_acc[d] = 0.0f;

        for (int jj = j; jj <= seg_end; ++jj) {
            size_t kv_offset = ((size_t)b * src_seq_len + jj) * kv_heads * head_dim + hk * head_dim;
            // 重新计算 score（因为未缓存）
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_cache[d] * static_cast<float>(k[kv_offset + d]);
            }
            score *= scale;
            float exp_val = expf(score - local_m);
            local_l += exp_val;
            // 累加 V（每个维度）
            for (int d = 0; d < head_dim; ++d) {
                float v_val = static_cast<float>(v[kv_offset + d]);
                local_acc[d] += exp_val * v_val;
            }
        }

        // 在线合并到全局
        float m_new = fmaxf(m, local_m);
        float alpha = expf(m - m_new);
        float beta = expf(local_m - m_new);
        l = l * alpha + local_l * beta;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * alpha + local_acc[d] * beta;
        }
        m = m_new;

        j = seg_end + 1;
    }

    // 写回
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        out[q_offset + d] = static_cast<T>(acc[d] * inv_l);
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

  dim3 block(32);
  dim3 grid(rows);
  
  size_t shared_mem_size=256*sizeof(float);
  rmsNormKernel<T><<<grid, block, shared_mem_size,stream>>>(
        input, weight, output,rows,hidden_dim, eps
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
    //     // 打印传入的维度参数（调试用）
    // printf("DEBUG Attention: batch=%d, target_seq=%d, src_seq=%d, q_heads=%d, kv_heads=%d, head_dim=%d, causal=%d\n",
    //        batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, (int)is_causal);
     // 计算元素总数
    int q_size = batch_size * target_seq_len * query_heads * head_dim;
    int k_size = batch_size * src_seq_len * kv_heads * head_dim;
    int v_size = k_size;
    int o_size = q_size;

    // 确保输出向量有足够的空间（可选，根据调用约定可调整）
    if (h_o.size() != o_size) {
        h_o.resize(o_size);
    }

    // 分配设备内存
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));

    // 拷贝输入数据到设备
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);

    // 缩放因子 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // 启动配置
    int total_queries = batch_size * target_seq_len * query_heads;
    int block_size = 256;
    int grid_size = (total_queries + block_size - 1) / block_size;  

    // 启动内核
    flash_attention_kernel<T><<<grid_size, block_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal, scale);

    // 拷贝输出回主机
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
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


