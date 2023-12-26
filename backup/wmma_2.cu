#include <chrono>
#include <cuda_fp16.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mma.h>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace nvcuda;
using namespace std;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
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

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
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

template <class T>
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

// Define the warp size and the matrix dimensions
#define WARP_SIZE 32

template <typename T>
__global__ void gemm_tensorcore(int m, int n, int k, const T* A, const T* B,
                                T* C)
{
    // Define the matrix tile size (K_TILE x M_TILE x N_TILE)
    const int K_TILE = 16;
    const int M_TILE = 16;
    const int N_TILE = 16;

    // Define the warp-level matrix multiply and accumulate fragment
    using FragmentA =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M_TILE, N_TILE, K_TILE,
                               half, nvcuda::wmma::row_major>;
    using FragmentB =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M_TILE, N_TILE, K_TILE,
                               half, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M_TILE,
                                             N_TILE, K_TILE, float>;

    // Calculate the block and thread indices
    int warpM = threadIdx.y / M_TILE;
    int warpN = threadIdx.x / N_TILE;

    // Define the warp-level fragments
    FragmentA fragA;
    FragmentB fragB;
    FragmentC fragC;

    // Initialize the warp-level accumulator fragment
    nvcuda::wmma::fill_fragment(fragC, 0.0f);

    // Loop over the K dimension
    for (int i = 0; i < k; i += K_TILE)
    {
        // Load matrix fragments from global memory to shared memory
        nvcuda::wmma::load_matrix_sync(
            fragA, &A[(warpM * M_TILE + threadIdx.y % M_TILE) * k + i], k);
        nvcuda::wmma::load_matrix_sync(
            fragB, &B[(warpN * N_TILE + threadIdx.x % N_TILE) + i * n], k);

        // Perform warp-level matrix multiply and accumulate
        nvcuda::wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store the warp-level accumulator fragment to global memory
    nvcuda::wmma::store_matrix_sync(
        &C[(warpM * M_TILE + threadIdx.y % M_TILE) * n +
           (warpN * N_TILE + threadIdx.x % N_TILE)],
        fragC, n, nvcuda::wmma::row_major);
}

// Create launch configuration for the kernel
template <typename T>
void launch_gemm_tensorcore(int m, int n, int k, const T* A, const T* B, T* C,
                            cudaStream_t stream)
{
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = WARP_SIZE;
    blockDim.y = WARP_SIZE;
    gridDim.x = (n + (blockDim.x * 2 - 1)) / (blockDim.x * 2);
    gridDim.y = (m + blockDim.y - 1) / blockDim.y;

    gemm_tensorcore<<<gridDim, blockDim, 0, stream>>>(m, n, k, A, B, C);
}

// Measure the performance of the kernel for 4096x4096x4096 matrices
template <typename T>
void measure_gemm_tensorcore(cudaStream_t stream)
{
    // Define the matrix dimensions
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Define the matrix sizes in bytes
    const int A_BYTES = M * K * sizeof(T);
    const int B_BYTES = K * N * sizeof(T);
    const int C_BYTES = M * N * sizeof(T);

    // Allocate memory for the input matrices
    T *A, *B, *C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&A, A_BYTES));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&B, B_BYTES));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&C, C_BYTES));

    // Create a random number generator
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<T> dist(-1.0f, 1.0f);

    // Initialize the input matrices
    std::vector<T> h_A(M * K);
    std::vector<T> h_B(K * N);
    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = dist(rng);
    }
    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = dist(rng);
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(A, h_A.data(), A_BYTES,
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(B, h_B.data(), B_BYTES,
                                     cudaMemcpyHostToDevice, stream));

    // Launch the kernel
    launch_gemm_tensorcore<T>(M, N, K, A, B, C, stream);

    // Measure the performance
    float latency = measure_performance<void*>(
        [=](cudaStream_t stream)
        { launch_gemm_tensorcore<T>(M, N, K, A, B, C, stream); },
        stream);

    // Print the performance
    std::cout << std::setw(8) << std::setprecision(4) << std::fixed << latency
              << " ms" << std::endl;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(A));
    CHECK_CUDA_ERROR(cudaFree(B));
    CHECK_CUDA_ERROR(cudaFree(C));
}

int main()
{
    // Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // // Measure the performance of the kernel for single-precision
    // floating-point std::cout << "Single-precision floating-point:" <<
    // std::endl; measure_gemm_tensorcore<float>(stream);

    // Measure the performance of the kernel for half-precision floating-point
    std::cout << "Half-precision floating-point:" << std::endl;
    measure_gemm_tensorcore<half>(stream);

    // Destroy the CUDA stream
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}
```