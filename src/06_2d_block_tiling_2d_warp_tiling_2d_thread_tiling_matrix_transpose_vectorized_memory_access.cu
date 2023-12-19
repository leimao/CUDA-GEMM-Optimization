#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v06.
// Each thread in the block processes THREAD_TILE_SIZE_Y *
// THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
          size_t NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v06_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{NUM_THREADS_X * NUM_THREADS_Y};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
        static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};

    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
    // THREAD_TILE_SIZE_X output values.
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
                          static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_Y{THREAD_TILE_SIZE_Y /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0U);

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K +
                         NUM_THREADS - 1U) /
                            NUM_THREADS;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) /
                VECTORIZED_BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) %
                VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};

            // These boundary checks might slow down the kernel to some extent.
            // But they guarantee the correctness of the kernel for all
            // different GEMM configurations.
            int4 A_row_vector_vals{0, 0, 0, 0};
            if (A_row_idx < m && A_col_idx < k)
            {
                A_row_vector_vals = *reinterpret_cast<int4 const*>(
                    &A[A_row_idx * lda + A_col_idx]);
            }
            if (A_col_idx + NUM_VECTOR_UNITS > k)
            {
                // Number of invalid elements in the last vector.
                size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS -
                                                  k};
                // Mask out the invalid elements.
                T* const A_row_vector_vals_ptr{
                    reinterpret_cast<T*>(&A_row_vector_vals)};
                for (size_t i{0U}; i < num_invalid_elements; ++i)
                {
                    A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                        static_cast<T>(0);
                }
            }
            // If this is true, the following if can be removed.
            // static_assert(VECTORIZED_BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y %
            // NUM_THREADS ==
            //               0U);
            if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
                A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
            {
                for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
                {
                    A_thread_block_tile_transposed
                        [A_thread_block_tile_col_idx + i]
                        [A_thread_block_tile_row_idx] =
                            reinterpret_cast<T const*>(&A_row_vector_vals)[i];
                }
            }
        }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X +
                         NUM_THREADS - 1U) /
                            NUM_THREADS;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) /
                VECTORIZED_BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (thread_linear_idx + load_idx * NUM_THREADS) %
                VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};

            // These boundary checks might slow down the kernel to some extent.
            // But they guarantee the correctness of the kernel for all
            // different GEMM configurations.
            int4 B_row_vector_vals{0, 0, 0, 0};
            if (B_row_idx < k && B_col_idx < n)
            {
                B_row_vector_vals = *reinterpret_cast<int4 const*>(
                    &B[B_row_idx * ldb + B_col_idx]);
            }
            if (B_col_idx + NUM_VECTOR_UNITS > n)
            {
                // Number of invalid elements in the last vector.
                size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS -
                                                  n};
                // Mask out the invalid elements.
                T* const B_row_vector_vals_ptr{
                    reinterpret_cast<T*>(&B_row_vector_vals)};
                for (size_t i{0U}; i < num_invalid_elements; ++i)
                {
                    B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                        static_cast<T>(0);
                }
            }
            // If this is true, the following if can be removed.
            // static_assert(VECTORIZED_BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %
            // NUM_THREADS ==
            //               0U);
            if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
                B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
            {
                *reinterpret_cast<int4*>(
                    &B_thread_block_tile[B_thread_block_tile_row_idx]
                                        [B_thread_block_tile_col_idx]) =
                    B_row_vector_vals;
            }
        }
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
// Can use pragma unroll to unroll these static loops to see if there is a
// performance gain.
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    thread_tile_repeat_row_idx *
                        (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
#pragma unroll
                for (size_t thread_tile_y_vector_idx{0U};
                     thread_tile_y_vector_idx < VECTORIZED_THREAD_TILE_SIZE_Y;
                     ++thread_tile_y_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &A_vals[thread_tile_repeat_row_idx]
                               [thread_tile_y_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &A_thread_block_tile_transposed
                                [A_thread_block_tile_col_idx]
                                [A_thread_block_tile_row_idx +
                                 thread_tile_y_vector_idx * NUM_VECTOR_UNITS]);
                }
            }
#pragma unroll
            for (size_t thread_tile_repeat_col_idx{0U};
                 thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_repeat_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_TILE_SIZE_X +
                    thread_tile_repeat_col_idx *
                        (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X};
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &B_vals[thread_tile_repeat_col_idx]
                               [thread_tile_x_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &B_thread_block_tile[B_thread_block_tile_row_idx]
                                                [B_thread_block_tile_col_idx +
                                                 thread_tile_x_vector_idx *
                                                     NUM_VECTOR_UNITS]);
                }
            }

// Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// products.
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_repeat_col_idx{0U};
                     thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_repeat_col_idx)
                {
#pragma unroll
                    for (size_t thread_tile_y_idx{0U};
                         thread_tile_y_idx < THREAD_TILE_SIZE_Y;
                         ++thread_tile_y_idx)
                    {
#pragma unroll
                        for (size_t thread_tile_x_idx{0U};
                             thread_tile_x_idx < THREAD_TILE_SIZE_X;
                             ++thread_tile_x_idx)
                        {
                            C_thread_results[thread_tile_repeat_row_idx]
                                            [thread_tile_repeat_col_idx]
                                            [thread_tile_y_idx]
                                            [thread_tile_x_idx] +=
                                A_vals[thread_tile_repeat_row_idx]
                                      [thread_tile_y_idx] *
                                B_vals[thread_tile_repeat_col_idx]
                                      [thread_tile_x_idx];
                        }
                    }
                }
            }
        }
        // We can use syncwarp now.
        __syncwarp();
    }
    // Need a synchronization before writing the results to DRAM.
    __syncthreads();

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U};
         thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_repeat_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U};
             thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_repeat_col_idx)
        {
#pragma unroll
            for (size_t thread_tile_y_idx{0U};
                 thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
            {
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_TILE_SIZE_Y +
                        thread_tile_repeat_row_idx *
                            (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y +
                        thread_tile_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_TILE_SIZE_X +
                        thread_tile_repeat_col_idx *
                            (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X +
                        thread_tile_x_vector_idx * NUM_VECTOR_UNITS};

                    if (C_row_idx < m && C_col_idx < n)
                    {
                        int4 C_vals{*reinterpret_cast<int4 const*>(
                            &C[C_row_idx * ldc + C_col_idx])};
#pragma unroll
                        for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
                        {
                            reinterpret_cast<T*>(&C_vals)[i] =
                                alpha *
                                    C_thread_results[thread_tile_repeat_row_idx]
                                                    [thread_tile_repeat_col_idx]
                                                    [thread_tile_y_idx]
                                                    [thread_tile_x_vector_idx *
                                                         NUM_VECTOR_UNITS +
                                                     i] +
                                beta * reinterpret_cast<T const*>(&C_vals)[i];
                        }
                        *reinterpret_cast<int4*>(
                            &C[C_row_idx * ldc + C_col_idx]) = C_vals;
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int WARP_TILE_SIZE_X{32U};
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v06_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                        NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v06_vectorized<float>(
    size_t m, size_t n, size_t k, float const* alpha, float const* A,
    size_t lda, float const* B, size_t ldb, float const* beta, float* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v06_vectorized<double>(
    size_t m, size_t n, size_t k, double const* alpha, double const* A,
    size_t lda, double const* B, size_t ldb, double const* beta, double* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v06_vectorized<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);