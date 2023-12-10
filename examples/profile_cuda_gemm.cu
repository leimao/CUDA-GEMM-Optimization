#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
void check_cublass(cublasStatus_t err, const char* const func,
                   const char* const file, const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << cublasGetStatusString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Determine CUDA data type from type.
template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
constexpr cudaDataType_t cuda_data_type_trait()
{
    if (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else if (std::is_same<T, __half>::value)
    {
        return CUDA_R_16F;
    }
    else
    {
        throw std::runtime_error("Unsupported data type.");
    }
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
void launch_gemm_cublas(size_t m, size_t n, size_t k, T const* alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T const* beta, T* C, size_t ldc, cublasHandle_t handle)
{
    // Non-TensorCore algorithm?
    constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
    constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};
    // All the matrix are in row-major order, non-transposed.
    // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
    CHECK_CUBLASS_ERROR(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, data_type, lda, B,
        data_type, ldb, beta, C, data_type, ldc, data_type, algo));
}

template <typename T>
bool all_close(T const* C, T const* C_ref, size_t m, size_t n, size_t ldc,
               T abs_tol)
{
    bool status{true};
    for (size_t i{0U}; i < n; ++i)
    {
        for (size_t j{0U}; i < n; ++i)
        {
            if (std::abs(C[i * ldc + j] - C_ref[i * ldc + j]) > abs_tol)
            {
                status = false;
                return status;
            }
        }
    }
    return status;
}

void print_device_info()
{
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;
}

template <typename T>
float compute_effective_bandwidth(size_t m, size_t n, size_t k, float latency)
{
    return ((m * k + k * n + m * n) * sizeof(T)) / (latency * 1e-3) / 1e9;
}

template <typename T>
float compute_effective_tflops(size_t m, size_t n, size_t k, float latency)
{
    return (2.0 * m * k * n) / (latency * 1e-3) / 1e12;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
void random_initialize_matrix(T* A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    auto const rand = [&dis, &gen]() { return dis(gen); };
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = rand();
        }
    }
}

int main()
{
    print_device_info();

    constexpr unsigned int num_repeats{10U};
    constexpr unsigned int num_warmups{10U};

    // constexpr size_t m{4096U};
    // constexpr size_t k{4096U};
    // constexpr size_t n{4096U};

    constexpr size_t m{2048U};
    constexpr size_t k{2048U};
    constexpr size_t n{2048U};

    // constexpr size_t lda{m};
    // constexpr size_t ldb{k};
    // constexpr size_t ldc{n};

    constexpr size_t lda{(m + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(m + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= n);
    static_assert(ldc >= n);

    constexpr float alpha{1.0f};
    constexpr float beta{0.0f};

    constexpr float abs_tol{1.0e-4f};

    std::cout << "Matrix Size: "
              << "M = " << m << " N = " << n << " K = " << k << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << std::endl;

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocate memory on host.
    float* A_host{nullptr};
    float* B_host{nullptr};
    float* C_host{nullptr};
    float* C_host_ref{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_ref, m * ldc * sizeof(float)));

    // Initialize matrix A and B.
    random_initialize_matrix(A_host, m, k, lda);
    random_initialize_matrix(B_host, k, n, ldb);
    random_initialize_matrix(C_host, m, n, ldc);

    // Allocate memory on device.
    float* A_device{nullptr};
    float* B_device{nullptr};
    float* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(float)));

    // Copy matrix A and B from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(float),
                                cudaMemcpyHostToDevice));

    // Create cuBLAS handle.
    cublasHandle_t handle;
    CHECK_CUBLASS_ERROR(cublasCreate(&handle));
    CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

    // Launch cuBLAS GEMM.
    float const latency_cublas{measure_performance<float>(
        [&](cudaStream_t stream)
        {
            launch_gemm_cublas<float>(m, n, k, &alpha, A_device, lda, B_device,
                                      ldb, &beta, C_device, ldc, handle);
            return 0.0f;
        },
        stream, num_repeats, num_warmups)};

    float const effective_bandwidth_cublas{
        compute_effective_bandwidth<float>(m, n, k, latency_cublas)};
    float const effective_tflops_cublas{
        compute_effective_tflops<float>(m, n, k, latency_cublas)};

    std::cout << "cuBLAS" << std::endl;
    std::cout << "Latency: " << latency_cublas << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth_cublas
              << " GB/s" << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops_cublas << " TFLOPS"
              << std::endl;
    std::cout << std::endl;

    // Copy matrix C from device to host.
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_device, m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));

    float* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(
        cudaMallocHost(&C_host_from_device, m * ldc * sizeof(float)));

    // Launch CUDA GEMM.
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(float),
                                cudaMemcpyHostToDevice));
    // Verify the correctness of CUDA GEMM.
    launch_gemm_kernel_v00<float>(m, n, k, &alpha, A_device, lda, B_device, ldb,
                                  &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(float),
                                cudaMemcpyDeviceToHost));
    assert(
        all_close<float>(C_host_from_device, C_host_ref, m, n, ldc, abs_tol));

    // Measure the performance of CUDA GEMM.
    float const latency_cuda_gemm_v00{measure_performance<float>(
        [&](cudaStream_t stream)
        {
            launch_gemm_kernel_v00<float>(m, n, k, &alpha, A_device, lda,
                                          B_device, ldb, &beta, C_device, ldc,
                                          stream);
            return 0.0f;
        },
        stream, num_repeats, num_warmups)};

    float const effective_bandwidth_cuda_gemm_v00{
        compute_effective_bandwidth<float>(m, n, k, latency_cuda_gemm_v00)};
    float const effective_tflops_cuda_gemm_v00{
        compute_effective_tflops<float>(m, n, k, latency_cuda_gemm_v00)};

    std::cout << "CUDA GEMM v00" << std::endl;
    std::cout << "Latency: " << latency_cuda_gemm_v00 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth_cuda_gemm_v00
              << " GB/s" << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops_cuda_gemm_v00
              << " TFLOPS" << std::endl;
    std::cout << std::endl;
}