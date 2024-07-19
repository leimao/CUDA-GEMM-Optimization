#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

#include <cuda_runtime.h>

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

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v02_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v03(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v03_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);
template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v04_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v05(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);
template <typename T>
void launch_gemm_kernel_v05_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v06(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);
template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v06_vectorized_double_buffered(
    size_t m, size_t n, size_t k, T const* alpha, T const* A, size_t lda,
    T const* B, size_t ldb, T const* beta, T* C, size_t ldc,
    cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v07(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v07_vectorized_double_buffered(
    size_t m, size_t n, size_t k, T const* alpha, T const* A, size_t lda,
    T const* B, size_t ldb, T const* beta, T* C, size_t ldc,
    cudaStream_t stream);
#endif