#ifndef CUDA_GEMM_UTILS_CUH
#define CUDA_GEMM_UTILS_CUH

#include <cuda_runtime.h>

#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_from_global_memory_to_shared_memory(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_Y = 0U>
__device__ void load_data_from_global_memory_to_shared_memory_transposed(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y +
                                                        BLOCK_TILE_SKEW_SIZE_Y],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // Removing the if will give another ~2 FLOPs performance on RTX
        // 3090. But it will make the kernel incorrect for some GEMM
        // configurations. T val{A[A_row_idx * lda + A_col_idx]}; This if
        // will slow down the kernel. Add static asserts from the host code
        // to guarantee this if is always true.
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                      [A_thread_block_tile_row_idx] = val;
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
                        NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // Removing the if will give another ~2 FLOPs performance on RTX
        // 3090. But it will make the kernel incorrect for some GEMM
        // configurations. T val{B[B_row_idx * ldb + B_col_idx]}; This if
        // will slow down the kernel. Add static asserts from the host code
        // to guarantee this if is always true.
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U, typename VECTOR_TYPE = int4>
__device__ void load_data_from_global_memory_to_shared_memory_vectorized(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile[BLOCK_TILE_SIZE_Y]
                         [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // The skew size could affect the data alignment in shared memory when we
    // use vectorized load. We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_K) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K) * sizeof(T) %
                      sizeof(VECTOR_TYPE) ==
                  0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) %
                      sizeof(VECTOR_TYPE) ==
                  0U);

// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
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
        VECTOR_TYPE A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k)
        {
            A_row_vector_vals = *reinterpret_cast<VECTOR_TYPE const*>(
                &A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS - k};
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
        // NUM_THREADS == 0U);
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
         load_idx <
         (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
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
        VECTOR_TYPE B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n)
        {
            B_row_vector_vals = *reinterpret_cast<VECTOR_TYPE const*>(
                &B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS - n};
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
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_Y = 0U, typename VECTOR_TYPE = int4>
__device__ void
load_data_from_global_memory_to_shared_memory_transposed_vectorized(
    T const* A, size_t lda, T const* B, size_t ldb,
    T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y +
                                                        BLOCK_TILE_SKEW_SIZE_Y],
    T B_thread_block_tile[BLOCK_TILE_SIZE_K]
                         [BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n,
    size_t k)
{
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // The skew size could affect the data alignment in shared memory when we
    // use vectorized load. We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) %
                      sizeof(VECTOR_TYPE) ==
                  0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) %
                      sizeof(VECTOR_TYPE) ==
                  0U);

// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
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
            A_row_vector_vals =
                *reinterpret_cast<int4 const*>(&A[A_row_idx * lda + A_col_idx]);
        }
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS - k};
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
                A_thread_block_tile_transposed[A_thread_block_tile_col_idx +
                                               i][A_thread_block_tile_row_idx] =
                    reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
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
            B_row_vector_vals =
                *reinterpret_cast<int4 const*>(&B[B_row_idx * ldb + B_col_idx]);
        }
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            // Number of invalid elements in the last vector.
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS - n};
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
}

#endif // CUDA_GEMM_UTILS_CUH