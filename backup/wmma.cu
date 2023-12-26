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

__global__ void test_wmma()
{
    // Warps within a block read from 256 byte aligned strided adresses to avoid
    // bank conflicts (makes no difference).
    __shared__ __half smem[1024 * 8];
    __half* A = smem + threadIdx.y * 1024 + threadIdx.y * 16;
    __half* B = smem + threadIdx.y * 1024 + threadIdx.y * 16 + 256;
    __half* C = smem + threadIdx.y * 1024 + threadIdx.y * 16 + 512;

    // Matrix A is read once, and accumulator is filled once.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);

#pragma unroll
    for (int i = 0; i < 20; i++)
    {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
            b_frag;
        wmma::load_matrix_sync(b_frag, B, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C, acc_frag, 16, wmma::mem_col_major);
}
// void TestWMMA()
// {
//         int threads = 256;
//         int blocks = 10000;
//         test_wmma<<<blocks, threads>>>();
// }
// int main(){

//         TestWMMA();
//         cudaDeviceSynchronize();
//         TestWMMA();
//         cudaDeviceSynchronize();
// }

void launch_wmma_test(cudaStream_t stream)
{
    int threads = 256;
    int blocks = 10000;
    test_wmma<<<blocks, threads, 0, stream>>>();
}

int main()
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> bound_function{
        std::bind(launch_wmma_test, std::placeholders::_1)};
    float const latency{measure_performance(bound_function, stream, 100, 100)};
    std::cout << "WMMA latency: " << latency << " ms" << std::endl;

    // Compute TFLOPS
    // Explain to the TFLOPS formula:
    // 1. 10000 is the number of blocks
    // 2. 16 is the number of threads per warp
    // 3. 16 is the number of warps per block
    // 4. 16 is the number of threads per block
    // 5. 2 is the number of FMA operations per thread

    float const tflops{10000.0f * 16.0f * 16.0f * 16.0f * 2.0f * 20 * 8 /
                       (latency * 1e-3f * 1e12f)};
    std::cout << "WMMA TFLOPS: " << tflops << std::endl;

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return 0;
}