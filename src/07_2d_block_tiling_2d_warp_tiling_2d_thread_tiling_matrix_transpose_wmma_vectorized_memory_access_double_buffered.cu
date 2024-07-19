#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://github.com/NVIDIA/cutlass/blob/b7508e337938137a699e486d8997646980acfc58/media/docs/programming_guidelines.md

template <
    typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
    size_t WMMA_TILE_SIZE_X, size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K,
    size_t NUM_WMMA_TILES_X, size_t NUM_WMMA_TILES_Y, size_t NUM_WMMA_TILES_K,
    size_t BLOCK_TILE_SKEW_SIZE_X, size_t BLOCK_TILE_SKEW_SIZE_Y>
__device__ void process_data_from_shared_memory_using_wmma(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y],
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X],
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X],
    T const A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K]
                                          [BLOCK_TILE_SIZE_Y +
                                           BLOCK_TILE_SKEW_SIZE_Y],
    T const B_thread_block_tile[BLOCK_TILE_SIZE_K]
                               [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t warp_row_idx, size_t warp_col_idx)
{
#pragma unroll
    for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
    {
#pragma unroll
        for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
             ++wmma_tile_row_idx)
        {
            nvcuda::wmma::load_matrix_sync(
                a_frags[wmma_tile_row_idx],
                &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K]
                                               [warp_row_idx *
                                                    WARP_TILE_SIZE_Y +
                                                wmma_tile_row_idx *
                                                    WMMA_TILE_SIZE_Y],
                BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
        }
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::load_matrix_sync(
                b_frags[wmma_tile_col_idx],
                &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K]
                                    [warp_col_idx * WARP_TILE_SIZE_X +
                                     wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);
        }
#pragma unroll
        for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
             ++wmma_tile_row_idx)
        {
#pragma unroll
            for (size_t wmma_tile_col_idx{0U};
                 wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
            {
                // Perform the matrix multiplication.
                nvcuda::wmma::mma_sync(
                    acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                    a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                    acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
            }
        }
    }
}

// GEMM kernel v07.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void
gemm_v07_vectorized_double_buffered(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    constexpr size_t NUM_PIPELINES{2U};
    // Only double buffer is supported in the implementation.
    // But even more number of pipelines can be supported if the implementation
    // is modified.
    static_assert(NUM_PIPELINES == 2U);
    static_assert((NUM_WARPS_X * NUM_WARPS_Y) % NUM_PIPELINES == 0U);
    static_assert(NUM_THREADS % NUM_PIPELINES == 0U);
    constexpr size_t NUM_THREADS_PER_PIPELINE{NUM_THREADS / NUM_PIPELINES};
    constexpr size_t NUM_WARPS_PER_PIPELINE{(NUM_WARPS_X * NUM_WARPS_Y) /
                                            NUM_PIPELINES};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T
        A_thread_block_tile_transposed[NUM_PIPELINES][BLOCK_TILE_SIZE_K]
                                      [BLOCK_TILE_SIZE_Y +
                                       BLOCK_TILE_SKEW_SIZE_Y];
    __shared__ T
        B_thread_block_tile[NUM_PIPELINES][BLOCK_TILE_SIZE_K]
                           [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        c_frag;

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    // Separate the warps to different pipelines.
    size_t const pipeline_index{warp_linear_idx / NUM_WARPS_PER_PIPELINE};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    if (pipeline_index == 0U)
    {
        // Pipeline 0 warps load buffer 0.
        load_data_from_global_memory_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS_PER_PIPELINE, BLOCK_TILE_SKEW_SIZE_X,
            BLOCK_TILE_SKEW_SIZE_Y>(
            A, lda, B, ldb, A_thread_block_tile_transposed[pipeline_index],
            B_thread_block_tile[pipeline_index], 0U,
            thread_linear_idx - pipeline_index * NUM_THREADS_PER_PIPELINE, m, n,
            k);
    }
    __syncthreads();

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         thread_block_tile_idx += NUM_PIPELINES)
    {
        if (pipeline_index == 0U)
        {
            // Pipeline 0 warps process buffer 0.
            process_data_from_shared_memory_using_wmma<
                T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, NUM_WMMA_TILES_X,
                NUM_WMMA_TILES_Y, NUM_WMMA_TILES_K, BLOCK_TILE_SKEW_SIZE_X,
                BLOCK_TILE_SKEW_SIZE_Y>(
                a_frags, b_frags, acc_frags,
                A_thread_block_tile_transposed[pipeline_index],
                B_thread_block_tile[pipeline_index], warp_row_idx,
                warp_col_idx);
            __syncthreads();

            // Pipeline 0 warps process buffer 1.
            if (thread_block_tile_idx + 1U < num_thread_block_tiles)
            {
                process_data_from_shared_memory_using_wmma<
                    T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                    WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                    WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, NUM_WMMA_TILES_X,
                    NUM_WMMA_TILES_Y, NUM_WMMA_TILES_K, BLOCK_TILE_SKEW_SIZE_X,
                    BLOCK_TILE_SKEW_SIZE_Y>(
                    a_frags, b_frags, acc_frags,
                    A_thread_block_tile_transposed[pipeline_index + 1],
                    B_thread_block_tile[pipeline_index + 1], warp_row_idx,
                    warp_col_idx);
            }
            __syncthreads();

            // Pipeline 0 warps load buffer 0.
            if (thread_block_tile_idx + 2U < num_thread_block_tiles)
            {
                load_data_from_global_memory_to_shared_memory_transposed_vectorized<
                    T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                    NUM_THREADS_PER_PIPELINE, BLOCK_TILE_SKEW_SIZE_X,
                    BLOCK_TILE_SKEW_SIZE_Y>(
                    A, lda, B, ldb,
                    A_thread_block_tile_transposed[pipeline_index],
                    B_thread_block_tile[pipeline_index],
                    thread_block_tile_idx + 2,
                    thread_linear_idx -
                        pipeline_index * NUM_THREADS_PER_PIPELINE,
                    m, n, k);
            }
            __syncthreads();
        }
        else
        {
            // Pipeline 1 warps load buffer 1.
            if (thread_block_tile_idx + 1U < num_thread_block_tiles)
            {
                load_data_from_global_memory_to_shared_memory_transposed_vectorized<
                    T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                    NUM_THREADS_PER_PIPELINE, BLOCK_TILE_SKEW_SIZE_X,
                    BLOCK_TILE_SKEW_SIZE_Y>(
                    A, lda, B, ldb,
                    A_thread_block_tile_transposed[pipeline_index],
                    B_thread_block_tile[pipeline_index],
                    thread_block_tile_idx + 1,
                    thread_linear_idx -
                        pipeline_index * NUM_THREADS_PER_PIPELINE,
                    m, n, k);
            }
            __syncthreads();

            // Pipeline 1 warps process buffer 0.
            process_data_from_shared_memory_using_wmma<
                T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, NUM_WMMA_TILES_X,
                NUM_WMMA_TILES_Y, NUM_WMMA_TILES_K, BLOCK_TILE_SKEW_SIZE_X,
                BLOCK_TILE_SKEW_SIZE_Y>(
                a_frags, b_frags, acc_frags,
                A_thread_block_tile_transposed[pipeline_index - 1],
                B_thread_block_tile[pipeline_index - 1], warp_row_idx,
                warp_col_idx);
            __syncthreads();

            // Pipeline 1 warps process buffer 1.
            if (thread_block_tile_idx + 1U < num_thread_block_tiles)
            {
                process_data_from_shared_memory_using_wmma<
                    T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                    WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                    WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K, NUM_WMMA_TILES_X,
                    NUM_WMMA_TILES_Y, NUM_WMMA_TILES_K, BLOCK_TILE_SKEW_SIZE_X,
                    BLOCK_TILE_SKEW_SIZE_Y>(
                    a_frags, b_frags, acc_frags,
                    A_thread_block_tile_transposed[pipeline_index],
                    B_thread_block_tile[pipeline_index], warp_row_idx,
                    warp_col_idx);
            }
            __syncthreads();
        }
    }

// Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // Load the fragment from global memory.
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                n, nvcuda::wmma::mem_row_major);
            // Perform scaling and addition.
            for (size_t i{0}; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] =
                    alpha *
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] +
                    beta * c_frag.x[i];
            }
            // Store the fragment back to global memory.
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename T>
void launch_gemm_kernel_v07_vectorized_double_buffered(
    size_t m, size_t n, size_t k, T const* alpha, T const* A, size_t lda,
    T const* B, size_t ldb, T const* beta, T* C, size_t ldc,
    cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    // The skew size is used to avoid bank conflicts in shared memory.
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v07_vectorized_double_buffered<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
        BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
        WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K,
        NUM_THREADS_PER_BLOCK><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v07_vectorized_double_buffered<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);