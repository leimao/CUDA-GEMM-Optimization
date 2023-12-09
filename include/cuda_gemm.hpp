#include "cuda_gemm_utils.hpp"

#include <cuda_fp16.h>

// template <typename T,
//           typename std::enable_if<std::is_same<T, float>::value ||
//                                       std::is_same<T, double>::value ||
//                                       std::is_same<T, __half>::value,
//                                   bool>::type = true>
// void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
//                             T const* A, size_t lda, T const* B, size_t ldb,
//                             T const* beta, T* C, size_t ldc,
//                             cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);
// void launch_gemm_kernel_v00(size_t m, size_t n, size_t k,
//                                              double const* alpha,
//                                              double const* A, size_t lda,
//                                              double const* B, size_t ldb,
//                                              double const* beta, double* C,
//                                              size_t ldc, cudaStream_t stream);
// void launch_gemm_kernel_v00(size_t m, size_t n, size_t k,
//                                              __half const* alpha,
//                                              __half const* A, size_t lda,
//                                              __half const* B, size_t ldb,
//                                              __half const* beta, __half* C,
//                                              size_t ldc, cudaStream_t stream);