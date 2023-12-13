#include <cassert>
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
    // All the matrix are in row-major order.
    // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
    // A: m x k row-major -> A: k x m column-major non-transposed
    // B: k x n row-major -> B: n x k column-major non-transposed
    // C: m x n row-major -> C: n x m column-major non-transposed
    // Thus, without padding, the leading dimension of the matrix in row-major
    // order is the number of columns, i.e., k for A, n for B, and n for C.
    // Row-major order: C = AB + C
    // Column-major order: C = BA + C
    // The cuBLAS API requires the leading dimension of the matrix in
    // column-major order. This API call looks non-intuitive, but it is correct.
    CHECK_CUBLASS_ERROR(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, data_type, ldb, A,
        data_type, lda, beta, C, data_type, ldc, data_type, algo));
}

template <typename T>
bool all_close(T const* C, T const* C_ref, size_t m, size_t n, size_t ldc,
               T abs_tol)
{
    bool status{true};
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            double const C_val{static_cast<double>(C[i * ldc + j])};
            double const C_ref_val{static_cast<double>(C_ref[i * ldc + j])};
            double const diff{C_val - C_ref_val};
            double const diff_val{std::abs(diff)};
            if (diff_val > static_cast<double>(abs_tol))
            {
                std::cout << "C[" << i << ", " << j << "] = " << C_val
                          << " C_ref[" << i << ", " << j << "] = " << C_ref_val
                          << std::endl;
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
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto const rand = [&dis, &gen]() { return dis(gen); };
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = static_cast<T>(rand());
        }
    }
}

void print_performance_result(size_t m, size_t n, size_t k, float latency)
{
    float const effective_bandwidth{
        compute_effective_bandwidth<float>(m, n, k, latency)};
    float const effective_tflops{compute_effective_tflops(m, n, k, latency)};

    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth << " GB/s"
              << std::endl;
    std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS"
              << std::endl;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
std::pair<float, float> profile_gemm(
    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc,
    std::function<void(size_t, size_t, size_t, T const*, T const*, size_t,
                       T const*, size_t, T const*, T*, size_t, cudaStream_t)>
        gemm_kernel_launch_function,
    size_t num_repeats = 10, size_t num_warmups = 10, unsigned int seed = 0U)
{
    T const alpha{static_cast<T>(1.0)};
    T const beta{static_cast<T>(0.0)};

    T const abs_tol{static_cast<T>(1.0e-2)};

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocate memory on host.
    T* A_host{nullptr};
    T* B_host{nullptr};
    T* C_host{nullptr};
    T* C_host_ref{nullptr};
    T* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_ref, m * ldc * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device, m * ldc * sizeof(T)));

    // Initialize matrix A and B.
    random_initialize_matrix(A_host, m, k, lda);
    random_initialize_matrix(B_host, k, n, ldb);
    random_initialize_matrix(C_host, m, n, ldc);

    // Allocate memory on device.
    T* A_device{nullptr};
    T* B_device{nullptr};
    T* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(T)));

    // Copy matrix A and B from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(T),
                                cudaMemcpyHostToDevice));

    // Create cuBLAS handle.
    cublasHandle_t handle;
    CHECK_CUBLASS_ERROR(cublasCreate(&handle));
    CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

    // Compute reference output.
    launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta,
                          C_device, ldc, handle);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy matrix C from device to host.
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_device, m * ldc * sizeof(T),
                                cudaMemcpyDeviceToHost));

    // Launch cuBLAS GEMM.
    float const latency_cublas{measure_performance<T>(
        [&](cudaStream_t stream)
        {
            launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb,
                                  &beta, C_device, ldc, handle);
            return 0.0f;
        },
        stream, num_repeats, num_warmups)};

    // Launch CUDA GEMM.
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(T),
                                cudaMemcpyHostToDevice));
    // Verify the correctness of CUDA GEMM.
    gemm_kernel_launch_function(m, n, k, &alpha, A_device, lda, B_device, ldb,
                                &beta, C_device, ldc, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device,
                                m * ldc * sizeof(T), cudaMemcpyDeviceToHost));
    assert(all_close<T>(C_host_from_device, C_host_ref, m, n, ldc, abs_tol));

    float const latency_cuda_gemm{measure_performance<T>(
        [&](cudaStream_t stream)
        {
            gemm_kernel_launch_function(m, n, k, &alpha, A_device, lda,
                                        B_device, ldb, &beta, C_device, ldc,
                                        stream);
            return 0.0f;
        },
        stream, num_repeats, num_warmups)};

    // Release resources.
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(B_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    CHECK_CUDA_ERROR(cudaFreeHost(A_host));
    CHECK_CUDA_ERROR(cudaFreeHost(B_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_ref));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
    CHECK_CUBLASS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    std::cout << "cuBLAS GEMM Kernel Performance" << std::endl;
    print_performance_result(m, n, k, latency_cublas);
    std::cout << "Custom GEMM Kernel Performance" << std::endl;
    print_performance_result(m, n, k, latency_cuda_gemm);
    std::cout << "Custom GEMM VS cuBLAS GEMM Performance: "
              << latency_cublas / latency_cuda_gemm * 100.0f << "%"
              << std::endl;

    return std::pair<float, float>{latency_cublas, latency_cuda_gemm};
}

int main()
{
    print_device_info();

    constexpr size_t num_repeats{20U};
    constexpr size_t num_warmups{20U};

    // constexpr size_t m{4096U};
    // constexpr size_t k{4096U};
    // constexpr size_t n{4096U};

    // constexpr size_t m{2048U};
    // constexpr size_t k{2048U};
    // constexpr size_t n{2048U};

    constexpr size_t m{1372U};
    constexpr size_t k{1153U};
    constexpr size_t n{2171U};

    // constexpr size_t lda{m};
    // constexpr size_t ldb{k};
    // constexpr size_t ldc{n};

    constexpr size_t lda{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(n + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= n);
    static_assert(ldc >= n);

    std::cout << "Matrix Size: "
              << "M = " << m << " N = " << n << " K = " << k << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << std::endl;

    // Define all the GEMM kernel launch functions to be profiled.
    std::vector<std::pair<
        std::string,
        std::function<void(size_t, size_t, size_t, float const*, float const*,
                           size_t, float const*, size_t, float const*, float*,
                           size_t, cudaStream_t)>>> const
        gemm_kernel_launch_functions{
            {"Custom GEMM Kernel V00", launch_gemm_kernel_v00<float>},
            {"Custom GEMM Kernel V01", launch_gemm_kernel_v01<float>},
            {"Custom GEMM Kernel V02", launch_gemm_kernel_v02<float>},
            {"Custom GEMM Kernel V02 Vectorized",
             launch_gemm_kernel_v02_vectorized<float>},
            {"Custom GEMM Kernel V03", launch_gemm_kernel_v03<float>},
            {"Custom GEMM Kernel V03 Vectorized",
             launch_gemm_kernel_v03_vectorized<float>},
            {"Custom GEMM Kernel V04", launch_gemm_kernel_v04<float>},
            {"Custom GEMM Kernel V04 Vectorized",
             launch_gemm_kernel_v04_vectorized<float>},
            {"Custom GEMM Kernel V05", launch_gemm_kernel_v05<float>},
            {"Custom GEMM Kernel V05 Vectorized",
             launch_gemm_kernel_v05_vectorized<float>}};

    for (auto const& gemm_kernel_launch_function : gemm_kernel_launch_functions)
    {
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> const gemm_kernel_profile_result{
            profile_gemm<float>(m, n, k, lda, ldb, ldc,
                                gemm_kernel_launch_function.second, num_repeats,
                                num_warmups)};
        std::cout << std::endl;
    }

    return 0;
}