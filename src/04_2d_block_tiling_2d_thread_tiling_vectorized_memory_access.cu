#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

// GEMM kernel v04.
// Coalesced read and write from global memory.
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v04_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    // B_vals is cached in the register.
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

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
                *reinterpret_cast<int4*>(
                    &A_thread_block_tile[A_thread_block_tile_row_idx]
                                        [A_thread_block_tile_col_idx]) =
                    A_row_vector_vals;
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

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                // There will be shared memory bank conflicts accessing the
                // values from A_thread_block_tile. We can do it better by
                // transposing the A_thread_block_tile when we load the data
                // from DRAM.
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile[A_thread_block_tile_row_idx +
                                        thread_tile_row_idx]
                                       [A_thread_block_tile_col_idx];
            }

            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X};
// Although the read from A_thread_block_tile cannot be vectorized, the read
// from B_thread_block_tile can be vectorized.
#pragma unroll
            for (size_t thread_tile_col_vector_idx{0U};
                 thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                 ++thread_tile_col_vector_idx)
            {
                *reinterpret_cast<int4*>(
                    &B_vals[thread_tile_col_vector_idx * NUM_VECTOR_UNITS]) =
                    *reinterpret_cast<int4 const*>(
                        &B_thread_block_tile[B_thread_block_tile_row_idx]
                                            [B_thread_block_tile_col_idx +
                                             thread_tile_col_vector_idx *
                                                 NUM_VECTOR_UNITS]);
            }

            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < THREAD_TILE_SIZE_X;
                     ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx]
                                    [thread_tile_col_idx] +=
                        A_vals[thread_tile_row_idx] *
                        B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // Vectorized writing the results to DRAM.
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_vector_idx{0U};
             thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
             ++thread_tile_col_vector_idx)
        {
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_Y +
                thread_tile_row_idx};
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X +
                thread_tile_col_vector_idx * NUM_VECTOR_UNITS};
            // Vectorized read from C.
            int4 C_row_vector_vals{*reinterpret_cast<int4 const*>(
                &C[C_row_idx * ldc + C_col_idx])};
            // Vectorized read from C_thread_results.
            int4 const C_thread_results_row_vector_vals{
                *reinterpret_cast<int4 const*>(
                    &C_thread_results[thread_tile_row_idx]
                                     [thread_tile_col_vector_idx *
                                      NUM_VECTOR_UNITS])};
            // Update the values in C_row_vector_vals
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                reinterpret_cast<T*>(&C_row_vector_vals)[i] =
                    alpha * reinterpret_cast<T const*>(
                                &C_thread_results_row_vector_vals)[i] +
                    beta * reinterpret_cast<T const*>(&C_row_vector_vals)[i];
            }
            // Vectorized write to C.
            if (C_row_idx < m && C_col_idx < n)
            {
                // No need to mask out the out-of-bound invalid elements,
                // because the row of C matrix is 32-byte aligned.
                *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]) =
                    C_row_vector_vals;
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v04_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // Feel free to play with the block tile sizes.
    // The algorithm correctness should always be guaranteed.
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    // Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v04_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                        THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit instantiation.
template void launch_gemm_kernel_v04_vectorized<float>(
    size_t m, size_t n, size_t k, float const* alpha, float const* A,
    size_t lda, float const* B, size_t ldb, float const* beta, float* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v04_vectorized<double>(
    size_t m, size_t n, size_t k, double const* alpha, double const* A,
    size_t lda, double const* B, size_t ldb, double const* beta, double* C,
    size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v04_vectorized<__half>(
    size_t m, size_t n, size_t k, __half const* alpha, __half const* A,
    size_t lda, __half const* B, size_t ldb, __half const* beta, __half* C,
    size_t ldc, cudaStream_t stream);