#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <iostream>

// #include <cublas_v2.h>
// #include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// #define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__,
// __LINE__) void check_cublass(cublasStatus_t err, const char* const func,
//                    const char* const file, const int line)
// {
//     if (err != CUBLAS_STATUS_SUCCESS)
//     {
//         std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
//         std::cerr << cublasGetStatusString(err) << std::endl;
//         std::exit(EXIT_FAILURE);
//     }
// }

// // Determine CUDA data type from type.
// template <typename T,
//           typename std::enable_if<std::is_same<T, float>::value ||
//                                       std::is_same<T, double>::value ||
//                                       std::is_same<T, __half>::value,
//                                   bool>::type = true>
// constexpr cudaDataType_t cuda_data_type_trait()
// {
//     if (std::is_same<T, float>::value)
//     {
//         return CUDA_R_32F;
//     }
//     else if (std::is_same<T, double>::value)
//     {
//         return CUDA_R_64F;
//     }
//     else if (std::is_same<T, __half>::value)
//     {
//         return CUDA_R_16F;
//     }
//     else
//     {
//         throw std::runtime_error("Unsupported data type.");
//     }
// }

// template <typename T,
//           typename std::enable_if<std::is_same<T, float>::value ||
//                                       std::is_same<T, double>::value ||
//                                       std::is_same<T, __half>::value,
//                                   bool>::type = true>
// void launch_gemm_cublas(size_t m, size_t n, size_t k, T const* alpha,
//                         T const* A, size_t lda, T const* B, size_t ldb,
//                         T const* beta, T* C, size_t ldc, cublasHandle_t
//                         handle)
// {
//     // Non-TensorCore algorithm?
//     constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
//     constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};
//     // All the matrix are in row-major order, non-transposed.
//     // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
//     CHECK_CUBLASS_ERROR(cublasGemmEx(
//         handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, data_type, lda,
//         B, data_type, ldb, beta, C, data_type, ldc, data_type, algo));
// }

#endif // CUDA_UTILS_HPP