#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

// #include "cuda_gemm_utils.hpp"

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

#endif