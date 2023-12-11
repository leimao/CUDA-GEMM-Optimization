#ifndef CUDA_GEMM_UTILS_HPP
#define CUDA_GEMM_UTILS_HPP

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line);

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line);

#endif // CUDA_GEMM_UTILS_HPP