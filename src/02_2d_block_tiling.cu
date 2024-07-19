#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v02.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    T sum{static_cast<T>(0)};
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_from_global_memory_to_shared_memory<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // Doing this results in 2 TOPS.
            // Suppose blockDim.x = blockDim.y = 32.
            // Effectively, for a warp, in one iteration, we read the value from
            // A_thread_block_tile at the same location on the shared memory
            // resulting in a broadcast, we also read 32 values that have no
            // bank conflicts from B_thread_block_tile. Even with that, all the
            // values have to be read from the shared memory and consequence is
            // the shared memory instruction runs very intensively just to
            // compute a small number of values using simple arithmetic
            // instructions, which is not efficient.
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v02<float>(size_t m, size_t n, size_t k,
                                            float const* alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const* beta,
                                            float* C, size_t ldc,
                                            cudaStream_t stream);
template void launch_gemm_kernel_v02<double>(size_t m, size_t n, size_t k,
                                             double const* alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const* beta, double* C,
                                             size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v02<__half>(size_t m, size_t n, size_t k,
                                             __half const* alpha,
                                             __half const* A, size_t lda,
                                             __half const* B, size_t ldb,
                                             __half const* beta, __half* C,
                                             size_t ldc, cudaStream_t stream);