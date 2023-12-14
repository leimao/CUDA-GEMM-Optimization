#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Refernece:
// https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
// https://siboehm.com/articles/22/CUDA-MMM

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cuda(T err, const char* const func, const char* const file,
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

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cublass(T err, const char* const func, const char* const file,
                   const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << err << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
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

template <typename T>
bool all_close(T const* arr_1, T const* arr_2, size_t n, T tol = 1e-6)
{
    bool status{true};
    for (size_t i{0U}; i < n; ++i)
    {
        if (std::abs(arr_1[i] - arr_2[i]) > tol)
        {
            std::cout << "arr_1[" << i << "] = " << arr_1[i] << std::endl;
            std::cout << "arr_2[" << i << "] = " << arr_2[i] << std::endl;
            status = false;
            return status;
        }
    }
    return status;
}

template <typename T>
void check_diff(T const* arr_1, T const* arr_2, size_t n)
{
    T error_abs_sum{0};
    T arr_1_abs_sum{0};
    T arr_2_abs_sum{0};
    for (size_t i{0U}; i < n; ++i)
    {
        error_abs_sum += std::abs(arr_1[i] - arr_2[i]);
        arr_1_abs_sum += std::abs(arr_1[i]);
        arr_2_abs_sum += std::abs(arr_2[i]);
    }
    std::cout << "Avg Abs Error: " << error_abs_sum / static_cast<float>(n)
              << std::endl;
    std::cout << "Avg Abs A1: " << arr_1_abs_sum / static_cast<float>(n)
              << std::endl;
    std::cout << "Avg Abs A2: " << arr_2_abs_sum / static_cast<float>(n)
              << std::endl;
}

template <typename T>
void init_random(T* arr, size_t n, T min = 0, T max = 1, unsigned int seed = 0)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(min, max);
    for (size_t i{0U}; i < n; ++i)
    {
        arr[i] = static_cast<T>(dis(gen));
    }
}

// Determine CUDA data type from type.
template <typename T>
constexpr cudaDataType_t cuda_data_type()
{
    if (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else
    {
        throw std::runtime_error("Unsupported data type.");
    }
}

// Use cuBLAS to compute GEMM.
// Input buffers are already on device memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_cublas(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C, cublasHandle_t handle)
{
    // Not 100% sure. This is non-TensorCore algorithm?
    // constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
    // Not 100% sure. This is TensorCore algorithm?
    constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT_TENSOR_OP};
    constexpr cudaDataType_t data_type{cuda_data_type<T>()};
    CHECK_CUBLASS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &alpha, A, data_type, k, B, data_type, n,
                                     &beta, C, data_type, n, data_type, algo));
}

// Use CPU to compute GEMM.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_cpu(size_t m, size_t n, size_t k, float alpha, T const* A,
                     T const* B, float beta, T* C)
{
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            T sum{0};
            for (size_t l{0U}; l < k; ++l)
            {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

// GEMM kernel v0.
// Non-coalesced read and write from global memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v0(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const m_i{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const n_i{blockIdx.y * blockDim.y + threadIdx.y};

    // Compute C[m_i, n_i] = alpha * A[m_i, :] * B[:, n_i] + beta * C[m_i, n_i].
    if (m_i < m && n_i < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_i{0U}; k_i < k; ++k_i)
        {
            sum += A[m_i * k + k_i] * B[k_i * n + n_i];
        }
        C[m_i * n + n_i] = alpha * sum + beta * C[m_i * n + n_i];
    }
}

// Launch GEMM v0 kernel.
// Input buffers are already on device memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v0(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v0<<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// GEMM kernel v1.
// Coalesced read and write from global memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v1(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const m_i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t const n_i{blockIdx.x * blockDim.x + threadIdx.x};

    // Compute C[m_i, n_i] = alpha * A[m_i, :] * B[:, n_i] + beta * C[m_i, n_i].
    if (m_i < m && n_i < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_i{0U}; k_i < k; ++k_i)
        {
            sum += A[m_i * k + k_i] * B[k_i * n + n_i];
        }
        C[m_i * n + n_i] = alpha * sum + beta * C[m_i * n + n_i];
    }
}

// Launch GEMM v1 kernel.
// Input buffers are already on device memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v1(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v1<<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// GEMM kernel v2.
// Coalesced read and write from global memory.
// Use shared memory to reduce the number of global memory accesses.
template <typename T, unsigned int BLOCK_SIZE,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v2(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const m_i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t const n_i{blockIdx.x * blockDim.x + threadIdx.x};

    // Here BLOCK_SIZE == WARP_SIZE.
    __shared__ T A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T B_tile[BLOCK_SIZE][BLOCK_SIZE];

    size_t const num_tiles{(k + BLOCK_SIZE - 1) / BLOCK_SIZE};

    // Compute C[m_i, n_i] = alpha * A[m_i, :] * B[:, n_i] + beta * C[m_i, n_i].
    T sum{static_cast<T>(0)};
    for (size_t tile_i{0U}; tile_i < num_tiles; ++tile_i)
    {
        size_t const k_i_A{tile_i * BLOCK_SIZE + threadIdx.x};
        // Load A[m_i, tile_i * BLOCK_SIZE : (tile_i + 1) * BLOCK_SIZE] into
        // A_tile.
        if (m_i < m && k_i_A < k)
        {
            // Coalesced read WARP_SIZE values per warp from global memory to
            // shared memory.
            A_tile[threadIdx.y][threadIdx.x] = A[m_i * k + k_i_A];
        }
        else
        {
            A_tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        size_t const k_i_B{tile_i * BLOCK_SIZE + threadIdx.y};
        // Load B[tile_i * BLOCK_SIZE : (tile_i + 1) * BLOCK_SIZE, n_i] into
        // B_tile.
        if (k_i_B < k && n_i < n)
        {
            // Coalesced read WARP_SIZE values per warp from global memory to
            // shared memory.
            B_tile[threadIdx.y][threadIdx.x] = B[k_i_B * n + n_i];
        }
        else
        {
            B_tile[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();
        // Compute the partial sum.
        // Effectively, for a warp, we read BLOCK_SIZE values from A_tile, and
        // BLOCK_SIZE * BLOCK_SIZE values from B_tile, just to compute
        // BLOCK_SIZE values of the partial sum. The consequence is the shared
        // memory instruction runs very intensively just to compute a small
        // number of values using simple arithmetic instructions, which is not
        // efficient.
        for (size_t k_i{0U}; k_i < BLOCK_SIZE; ++k_i)
        {
            // Doing this reulst in 2 TOPS.
            sum += A_tile[threadIdx.y][k_i] * B_tile[k_i][threadIdx.x];
            // Doing this results in 40 TOPS, whic is almost the speed of light.
            // sum += 0U;
        }
        __syncthreads();
    }
    if (m_i < m && n_i < n)
    {
        C[m_i * n + n_i] = alpha * sum + beta * C[m_i * n + n_i];
    }
}

// Launch GEMM v2 kernel.
// Input buffers are already on device memory.
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v2(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    constexpr unsigned int BLOCK_SIZE{32U};
    dim3 const block_dim{BLOCK_SIZE, BLOCK_SIZE, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v2<T, BLOCK_SIZE>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// GEMM kernel v2 utilizes shared memory to reduce the number of global memory
// accesses.
template <typename T, unsigned int BLOCK_TILE_SIZE_X,
          unsigned int BLOCK_TILE_SIZE_Y, unsigned int BLOCK_TILE_SIZE_K,
          unsigned int NUM_WARPS_PER_BLOCK_X,
          unsigned int NUM_WARPS_PER_BLOCK_Y,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v3(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    size_t const warp_idx{threadIdx.y};
    size_t const thread_idx_linear{threadIdx.y * blockDim.x + threadIdx.x};
    constexpr size_t num_threads{NUM_WARPS_PER_BLOCK_Y * NUM_WARPS_PER_BLOCK_X *
                                 32U};
    // constexpr size_t num_C_results{BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
    // num_threads};

    constexpr unsigned int WARP_THREAD_SIZE_X{4U};
    constexpr unsigned int WARP_THREAD_SIZE_Y{8U};

    // BLOCK_TILE_SIZE_X = 128
    // BLOCK_TILE_SIZE_Y = 128
    // BLOCK_TILE_SIZE_K = 32
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // Each thread block computes a thread block tile of C of size BLOCK_SIZE_Y
    // * BLOCK_SIZE_X.
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};

    // constexpr size_t A_fragment_size{BLOCK_TILE_SIZE_Y /
    // NUM_WARPS_PER_BLOCK_Y / WARP_THREAD_SIZE_Y}; constexpr size_t
    // B_fragment_size{BLOCK_TILE_SIZE_X / NUM_WARPS_PER_BLOCK_X /
    // WARP_THREAD_SIZE_X}; constexpr size_t C_fragment_size{A_fragment_size *
    // B_fragment_size}; static_assert(A_fragment_size == 8U);
    // static_assert(B_fragment_size == 8U);
    // T A_fragment[A_fragment_size];
    // T B_fragment[B_fragment_size];
    // T C_fragment[C_fragment_size];
    T C_results[BLOCK_TILE_SIZE_X] = {static_cast<T>(0)};

    for (size_t tile_i{0U}; tile_i < num_thread_block_tiles; ++tile_i)
    {
        // Load data from A to shared memory A_thread_block_tile.
        // Each thread loads BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K / (blockDim.x
        // * blockDim.y) values. Each warp loads BLOCK_TILE_SIZE_K values to one
        // row of A_thread_block_tile.
        for (size_t row_idx_A_thread_block_tile{warp_idx};
             row_idx_A_thread_block_tile < BLOCK_TILE_SIZE_Y;
             row_idx_A_thread_block_tile += blockDim.y)
        {
            size_t const row_idx_A{row_idx_A_thread_block_tile +
                                   blockIdx.y * BLOCK_TILE_SIZE_Y};
            size_t const col_idx_A{tile_i * BLOCK_TILE_SIZE_K + threadIdx.x};
            if (row_idx_A < m && col_idx_A < k)
            {
                A_thread_block_tile[row_idx_A_thread_block_tile][threadIdx.x] =
                    A[row_idx_A * k + col_idx_A];
            }
            else
            {
                A_thread_block_tile[row_idx_A_thread_block_tile][threadIdx.x] =
                    static_cast<T>(0);
            }
        }
        // Load data from B to shared memory B_thread_block_tile.
        // Each thread loads BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / (blockDim.x
        // * blockDim.y) values.
        for (size_t linear_idx_B_thread_block_tile{thread_idx_linear};
             linear_idx_B_thread_block_tile <
             BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X;
             linear_idx_B_thread_block_tile += blockDim.x * blockDim.y)
        {
            size_t const row_idx_B_thread_block_tile{
                linear_idx_B_thread_block_tile / BLOCK_TILE_SIZE_X};
            size_t const col_idx_B_thread_block_tile{
                linear_idx_B_thread_block_tile % BLOCK_TILE_SIZE_X};
            size_t const row_idx_B{tile_i * BLOCK_TILE_SIZE_K +
                                   row_idx_B_thread_block_tile};
            size_t const col_idx_B{col_idx_B_thread_block_tile +
                                   blockIdx.x * BLOCK_TILE_SIZE_X};
            if (row_idx_B < k && col_idx_B < n)
            {
                B_thread_block_tile[row_idx_B_thread_block_tile]
                                   [col_idx_B_thread_block_tile] =
                                       B[row_idx_B * n + col_idx_B];
            }
            else
            {
                B_thread_block_tile[row_idx_B_thread_block_tile]
                                   [col_idx_B_thread_block_tile] =
                                       static_cast<T>(0);
            }
        }
        __syncthreads();

        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // Cache data from shared memory to registers to alleviate the
            // shared memory access pressure.
            size_t const row_idx_B_thread_block_tile{k_i};
            // Assert blockDim.x * blockDim.y >= BLOCK_TILE_SIZE_X
            // This wasted half of the threads in the thread block.
            size_t const col_idx_B_thread_block_tile{thread_idx_linear %
                                                     BLOCK_TILE_SIZE_X};
            // Cache the data to register.
            T const cache_B{B_thread_block_tile[row_idx_B_thread_block_tile]
                                               [col_idx_B_thread_block_tile]};

            for (size_t idx_C_results{0U}; idx_C_results < BLOCK_TILE_SIZE_X;
                 ++idx_C_results)
            {
                size_t const row_idx_A_thread_block{idx_C_results};
                size_t const col_idx_A_thread_block{k_i};
                T const val_A{A_thread_block_tile[row_idx_A_thread_block]
                                                 [col_idx_A_thread_block]};
                C_results[idx_C_results] += cache_B * val_A;
            }
        }
    }
    for (size_t idx_C_results{0U}; idx_C_results < BLOCK_TILE_SIZE_X;
         ++idx_C_results)
    {
        size_t const row_idx_C_thread_block_tile{
            idx_C_results + blockIdx.y * BLOCK_TILE_SIZE_Y};
        size_t const col_idx_C_thread_block_tile{
            thread_idx_linear % BLOCK_TILE_SIZE_X +
            blockIdx.x * BLOCK_TILE_SIZE_X};
        if (thread_idx_linear / BLOCK_TILE_SIZE_Y == 0 &&
            row_idx_C_thread_block_tile < m && col_idx_C_thread_block_tile < n)
        {
            C[row_idx_C_thread_block_tile * n + col_idx_C_thread_block_tile] =
                alpha * C_results[idx_C_results] +
                beta * C[row_idx_C_thread_block_tile * n +
                         col_idx_C_thread_block_tile];
        }
    }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v3(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    constexpr unsigned int WARP_NUM_THREADS{32U};
    constexpr unsigned int NUM_WARPS_PER_BLOCK_X{4U};
    constexpr unsigned int NUM_WARPS_PER_BLOCK_Y{2U};
    constexpr unsigned int NUM_WARPS_PER_BLOCK{NUM_WARPS_PER_BLOCK_Y *
                                               NUM_WARPS_PER_BLOCK_X};
    constexpr unsigned int BLOCK_TILE_SIZE_X{WARP_NUM_THREADS *
                                             NUM_WARPS_PER_BLOCK_X};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{BLOCK_TILE_SIZE_X};
    constexpr unsigned int BLOCK_TILE_SIZE_K{WARP_NUM_THREADS};
    dim3 const block_dim{WARP_NUM_THREADS, NUM_WARPS_PER_BLOCK, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v3<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_WARPS_PER_BLOCK_X, NUM_WARPS_PER_BLOCK_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// 1D thread tiling
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y output values.
// Number of threads BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X /
// BLOCK_FRAGMENT_SIZE_Y
template <typename T, unsigned int BLOCK_TILE_SIZE_X,
          unsigned int BLOCK_TILE_SIZE_Y, unsigned int BLOCK_TILE_SIZE_K,
          unsigned int BLOCK_FRAGMENT_SIZE_Y,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v5(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_threads{blockDim.x};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // BLOCK_FRAGMENT_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * BLOCK_FRAGMENT_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    T C_thread_results[BLOCK_FRAGMENT_SIZE_Y] = {static_cast<T>(0)};

    size_t const B_thread_block_tile_col_idx_cached{threadIdx.x %
                                                    BLOCK_TILE_SIZE_X};
    size_t const A_thread_block_tile_col_idx_cached{threadIdx.x /
                                                    BLOCK_TILE_SIZE_X};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // Load data from A on DRAM to A_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K / num_threads;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};

            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels.
            T const val{(A_row_idx < m && A_col_idx < k)
                            ? A[A_row_idx * k + A_col_idx]
                            : static_cast<T>(0)};
            A_thread_block_tile[A_thread_block_tile_row_idx]
                               [A_thread_block_tile_col_idx] = val;

            // A_thread_block_tile[A_thread_block_tile_row_idx]
            //                    [A_thread_block_tile_col_idx] =
            //                        A[A_row_idx * k + A_col_idx];
        }
        // Load data from B on DRAM to B_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_X};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};

            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels.
            T const val{(B_row_idx < k && B_col_idx < n)
                            ? B[B_row_idx * n + B_col_idx]
                            : static_cast<T>(0)};
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx] = val;
            // B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx] =
            //                        B[B_row_idx * n + B_col_idx];
        }
        __syncthreads();

        // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
        // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
        // shared memory as A_thread_block_tile and B_thread_block_tile,
        // respectively. This inner product is further decomposed to
        // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
        // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
        // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
        // B_thread_block_tile can be cached in the register.
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // B_val is from the k_i row of the B_thread_block_tile.
            size_t const B_thread_block_tile_row_idx{k_i};
            // size_t const B_thread_block_tile_col_idx{threadIdx.x %
            //                                          BLOCK_TILE_SIZE_X};
            // B_val is cached in the register.
            T const B_val{
                B_thread_block_tile[B_thread_block_tile_row_idx]
                                   [B_thread_block_tile_col_idx_cached]};
            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    A_thread_block_tile_col_idx_cached * BLOCK_FRAGMENT_SIZE_Y +
                    fragment_y_idx};
                size_t const A_thread_block_tile_col_idx{k_i};
                T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx]
                                                 [A_thread_block_tile_col_idx]};
                C_thread_results[fragment_y_idx] += B_val * A_val;
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t fragment_y_idx{0U}; fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y;
         ++fragment_y_idx)
    {
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               threadIdx.x / BLOCK_TILE_SIZE_X *
                                   BLOCK_FRAGMENT_SIZE_Y +
                               fragment_y_idx};
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               threadIdx.x % BLOCK_TILE_SIZE_X};
        if (C_row_idx < m && C_col_idx < n)
        {
            C[C_row_idx * n + C_col_idx] =
                alpha * C_thread_results[fragment_y_idx] +
                beta * C[C_row_idx * n + C_col_idx];
        }
    }
}

// 1D thread tiling
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y output values.
// Number of threads BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X /
// BLOCK_FRAGMENT_SIZE_Y
template <typename T, unsigned int BLOCK_TILE_SIZE_X,
          unsigned int BLOCK_TILE_SIZE_Y, unsigned int BLOCK_TILE_SIZE_K,
          unsigned int BLOCK_FRAGMENT_SIZE_Y,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v5_1(size_t m, size_t n, size_t k, float alpha, T const* A,
                          T const* B, float beta, T* C)
{
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // If using blockDim.x as a variable for num_threads, the performance
    // dropped by 2 TOPS, which is 25% of the total performance! size_t const
    // num_threads{blockDim.x};
    constexpr size_t num_threads{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 BLOCK_FRAGMENT_SIZE_Y};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // BLOCK_FRAGMENT_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * BLOCK_FRAGMENT_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    T C_thread_results[BLOCK_FRAGMENT_SIZE_Y] = {static_cast<T>(0)};

    size_t const B_thread_block_tile_col_idx_cached{threadIdx.x %
                                                    BLOCK_TILE_SIZE_X};
    size_t const A_thread_block_tile_col_idx_cached{threadIdx.x /
                                                    BLOCK_TILE_SIZE_X};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // Load data from A on DRAM to A_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K / num_threads;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};

            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels.
            T const val{(A_row_idx < m && A_col_idx < k)
                            ? A[A_row_idx * k + A_col_idx]
                            : static_cast<T>(0)};
            A_thread_block_tile[A_thread_block_tile_row_idx]
                               [A_thread_block_tile_col_idx] = val;

            // A_thread_block_tile[A_thread_block_tile_row_idx]
            //                    [A_thread_block_tile_col_idx] =
            //                        A[A_row_idx * k + A_col_idx];
        }
        // Load data from B on DRAM to B_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_X};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};

            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. Normally, the CUDA examples showing good performance on
            // the internet will not do this. Therefore, the correctness of
            // their kernel is not guaranteed for all the GEMM configurations (m
            // x n x k). As a general purpose kernel, you cannot claim
            // performance without ensuring the correctness of all the corner
            // cases.
            T const val{(B_row_idx < k && B_col_idx < n)
                            ? B[B_row_idx * n + B_col_idx]
                            : static_cast<T>(0)};
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx] = val;
            // B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx] =
            //                        B[B_row_idx * n + B_col_idx];
        }
        __syncthreads();

        // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
        // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
        // shared memory as A_thread_block_tile and B_thread_block_tile,
        // respectively. This inner product is further decomposed to
        // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
        // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
        // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
        // B_thread_block_tile can be cached in the register.
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // B_val is from the k_i row of the B_thread_block_tile.
            size_t const B_thread_block_tile_row_idx{k_i};
            // size_t const B_thread_block_tile_col_idx{threadIdx.x %
            //                                          BLOCK_TILE_SIZE_X};
            // B_val is cached in the register.
            T const B_val{
                B_thread_block_tile[B_thread_block_tile_row_idx]
                                   [B_thread_block_tile_col_idx_cached]};
            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    A_thread_block_tile_col_idx_cached * BLOCK_FRAGMENT_SIZE_Y +
                    fragment_y_idx};
                size_t const A_thread_block_tile_col_idx{k_i};
                T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx]
                                                 [A_thread_block_tile_col_idx]};
                C_thread_results[fragment_y_idx] += B_val * A_val;
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t fragment_y_idx{0U}; fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y;
         ++fragment_y_idx)
    {
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               threadIdx.x / BLOCK_TILE_SIZE_X *
                                   BLOCK_FRAGMENT_SIZE_Y +
                               fragment_y_idx};
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               threadIdx.x % BLOCK_TILE_SIZE_X};
        if (C_row_idx < m && C_col_idx < n)
        {
            C[C_row_idx * n + C_col_idx] =
                alpha * C_thread_results[fragment_y_idx] +
                beta * C[C_row_idx * n + C_col_idx];
        }
    }
}

template <typename T, const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, T alpha, const T* A,
                                   const T* B, T beta, T* C)
{
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of
    // A. The slower configuration would share columns of A, but access into B
    // would be non-sequential. So the faster configuration has better spatial
    // locality and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    T threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            T tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx)
            {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx)
    {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_siboehm_v5(size_t m, size_t n, size_t k, float alpha,
                                   T const* A, T const* B, float beta, T* C,
                                   cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread in the block will process BLOCK_FRAGMENT_SIZE_Y elements.
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{8U};
    // constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{32U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / BLOCK_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % BLOCK_FRAGMENT_SIZE_Y == 0U);
    // These are for DRAM coalesced read/write.
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    // gemm_v5<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
    //         BLOCK_FRAGMENT_SIZE_Y>
    //     <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    // gemm_v5_1<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
    //         BLOCK_FRAGMENT_SIZE_Y>
    //     <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    sgemm1DBlocktiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                       BLOCK_TILE_SIZE_K, BLOCK_FRAGMENT_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);

    CHECK_LAST_CUDA_ERROR();
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v5(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread in the block will process BLOCK_FRAGMENT_SIZE_Y elements.
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{8U};
    // constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{32U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / BLOCK_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % BLOCK_FRAGMENT_SIZE_Y == 0U);
    // These are for DRAM coalesced read/write.
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    // gemm_v5<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
    //         BLOCK_FRAGMENT_SIZE_Y>
    //     <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    gemm_v5_1<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
              BLOCK_FRAGMENT_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    // sgemm1DBlocktiling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
    // BLOCK_TILE_SIZE_K, BLOCK_FRAGMENT_SIZE_Y>
    //     <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);

    CHECK_LAST_CUDA_ERROR();
}

// 2D thread tiling
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <typename T, unsigned int BLOCK_TILE_SIZE_X,
          unsigned int BLOCK_TILE_SIZE_Y, unsigned int BLOCK_TILE_SIZE_K,
          unsigned int BLOCK_FRAGMENT_SIZE_X,
          unsigned int BLOCK_FRAGMENT_SIZE_Y,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void
// Launch bound slightly improves the performance of the 2D block tiling
// implementation.
__launch_bounds__(BLOCK_TILE_SIZE_Y* BLOCK_TILE_SIZE_X /
                  (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X))
    gemm_v6(size_t m, size_t n, size_t k, float alpha, T const* A, T const* B,
            float beta, T* C)
{
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // B_vals is cached in the register.
    T B_vals[BLOCK_FRAGMENT_SIZE_X] = {static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[BLOCK_FRAGMENT_SIZE_Y] = {static_cast<T>(0)};

    constexpr unsigned int num_threads{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (BLOCK_FRAGMENT_SIZE_X * BLOCK_FRAGMENT_SIZE_Y)};
    // size_t const num_threads{blockDim.x};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // BLOCK_FRAGMENT_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * BLOCK_FRAGMENT_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // BLOCK_FRAGMENT_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * BLOCK_FRAGMENT_SIZE_X]
    T C_thread_results[BLOCK_FRAGMENT_SIZE_Y][BLOCK_FRAGMENT_SIZE_X] = {
        static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // Load data from A on DRAM to A_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K / num_threads;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};
            A_thread_block_tile[A_thread_block_tile_row_idx]
                               [A_thread_block_tile_col_idx] =
                                   A[A_row_idx * k + A_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. T const val{(A_row_idx < m && A_col_idx < k) ?
            // A[A_row_idx * k + A_col_idx] : static_cast<T>(0)};
            // A_thread_block_tile[A_thread_block_tile_row_idx]
            //                    [A_thread_block_tile_col_idx] = val;
        }
        // Load data from B on DRAM to B_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_X};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx] =
                                   B[B_row_idx * n + B_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. T const val{(B_row_idx < k && B_col_idx < n) ?
            // B[B_row_idx * n + B_col_idx] : static_cast<T>(0)};
            // B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx] = val;
        }
        __syncthreads();

        // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
        // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
        // shared memory as A_thread_block_tile and B_thread_block_tile,
        // respectively. This inner product is further decomposed to
        // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
        // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
        // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
        // both A_thread_block_tile and B_thread_block_tile can be cached in the
        // register.
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // B_vals is from the k_i row of the B_thread_block_tile.
            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                threadIdx.x % (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                BLOCK_FRAGMENT_SIZE_X};

            for (size_t fragment_x_idx{0U};
                 fragment_x_idx < BLOCK_FRAGMENT_SIZE_X; ++fragment_x_idx)
            {
                B_vals[fragment_x_idx] =
                    B_thread_block_tile[B_thread_block_tile_row_idx]
                                       [B_thread_block_tile_col_idx +
                                        fragment_x_idx];
            }
            // A_vals is from the k_i column of the A_thread_block_tile.
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x / (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X)) *
                BLOCK_FRAGMENT_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                // There will be shared memory bank conflicts accessing the
                // values from A_thread_block_tile. We can do it better by
                // transposing the A_thread_block_tile when we load the data
                // from DRAM.
                A_vals[fragment_y_idx] =
                    A_thread_block_tile[A_thread_block_tile_row_idx +
                                        fragment_y_idx]
                                       [A_thread_block_tile_col_idx];
            }

            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                for (size_t fragment_x_idx{0U};
                     fragment_x_idx < BLOCK_FRAGMENT_SIZE_X; ++fragment_x_idx)
                {
                    C_thread_results[fragment_y_idx][fragment_x_idx] +=
                        A_vals[fragment_y_idx] * B_vals[fragment_x_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t fragment_y_idx{0U}; fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y;
         ++fragment_y_idx)
    {
        for (size_t fragment_x_idx{0U}; fragment_x_idx < BLOCK_FRAGMENT_SIZE_X;
             ++fragment_x_idx)
        {
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                threadIdx.x / (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                    BLOCK_FRAGMENT_SIZE_Y +
                fragment_y_idx};
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                threadIdx.x % (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                    BLOCK_FRAGMENT_SIZE_X +
                fragment_x_idx};
            if (C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * n + C_col_idx] =
                    alpha * C_thread_results[fragment_y_idx][fragment_x_idx] +
                    beta * C[C_row_idx * n + C_col_idx];
            }
        }
    }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v6(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    // constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread in the block will process BLOCK_FRAGMENT_SIZE_Y *
    // BLOCK_FRAGMENT_SIZE_Y elements.
    // constexpr unsigned int BLOCK_FRAGMENT_SIZE_X{4U};
    // constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{4U};
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_X{8U};
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (BLOCK_FRAGMENT_SIZE_X * BLOCK_FRAGMENT_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % BLOCK_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % BLOCK_FRAGMENT_SIZE_Y == 0U);
    // These are for DRAM coalesced read/write.
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v6<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            BLOCK_FRAGMENT_SIZE_X, BLOCK_FRAGMENT_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// 2D block tiling with transposed A_thread_block_tile
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <typename T, unsigned int BLOCK_TILE_SIZE_X,
          unsigned int BLOCK_TILE_SIZE_Y, unsigned int BLOCK_TILE_SIZE_K,
          unsigned int BLOCK_FRAGMENT_SIZE_X,
          unsigned int BLOCK_FRAGMENT_SIZE_Y,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void
// Launch bound slightly improves the performance of the 2D block tiling
// implementation.
__launch_bounds__(BLOCK_TILE_SIZE_Y* BLOCK_TILE_SIZE_X /
                  (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X))
    gemm_v7(size_t m, size_t n, size_t k, float alpha, T const* A, T const* B,
            float beta, T* C)
{
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // B_vals is cached in the register.
    T B_vals[BLOCK_FRAGMENT_SIZE_X] = {static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[BLOCK_FRAGMENT_SIZE_Y] = {static_cast<T>(0)};

    size_t const num_threads{blockDim.x};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
    // BLOCK_FRAGMENT_SIZE_X output values. Specifically, these values
    // corresponds to C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x /
    // BLOCK_TILE_SIZE_X * BLOCK_FRAGMENT_SIZE_Y : blockIdx.y *
    // BLOCK_TILE_SIZE_Y + (threadIdx.x / BLOCK_TILE_SIZE_X + 1) *
    // BLOCK_FRAGMENT_SIZE_Y][blockIdx.x * BLOCK_TILE_SIZE_X + threadIdx.x %
    // BLOCK_TILE_SIZE_X * BLOCK_FRAGMENT_SIZE_X : blockIdx.x *
    // BLOCK_TILE_SIZE_X + (threadIdx.x % BLOCK_TILE_SIZE_X + 1) *
    // BLOCK_FRAGMENT_SIZE_X]
    T C_thread_results[BLOCK_FRAGMENT_SIZE_Y][BLOCK_FRAGMENT_SIZE_X] = {
        static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // Load data from A on DRAM to A_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K / num_threads;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};
            // A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]
            // = A[A_row_idx * k + A_col_idx];
            // A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //                               [A_thread_block_tile_row_idx] =
            //                                   A[A_row_idx * k + A_col_idx];
            T const val{(A_row_idx < m && A_col_idx < k)
                            ? A[A_row_idx * k + A_col_idx]
                            : static_cast<T>(0)};
            A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                          [A_thread_block_tile_row_idx] = val;
        }
        // Load data from B on DRAM to B_thread_block_tile on shared memory.
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * num_threads) / BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * num_threads) % BLOCK_TILE_SIZE_X};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};
            // B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx] =
            //                        B[B_row_idx * n + B_col_idx];
            T const val{(B_row_idx < k && B_col_idx < n)
                            ? B[B_row_idx * n + B_col_idx]
                            : static_cast<T>(0)};
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx] = val;
        }
        __syncthreads();

        // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
        // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
        // shared memory as A_thread_block_tile and B_thread_block_tile,
        // respectively. This inner product is further decomposed to
        // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
        // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
        // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
        // both A_thread_block_tile and B_thread_block_tile can be cached in the
        // register.
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // B_vals is from the k_i row of the B_thread_block_tile.
            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                threadIdx.x % (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                BLOCK_FRAGMENT_SIZE_X};

            for (size_t fragment_x_idx{0U};
                 fragment_x_idx < BLOCK_FRAGMENT_SIZE_X; ++fragment_x_idx)
            {
                B_vals[fragment_x_idx] =
                    B_thread_block_tile[B_thread_block_tile_row_idx]
                                       [B_thread_block_tile_col_idx +
                                        fragment_x_idx];
            }
            // A_vals is from the k_i column of the A_thread_block_tile.
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x / (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X)) *
                BLOCK_FRAGMENT_SIZE_Y};
            size_t const A_thread_block_tile_col_idx{k_i};

            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                // There will be shared memory bank conflicts accessing the
                // values from A_thread_block_tile. We can do it better by
                // transposing the A_thread_block_tile when we load the data
                // from DRAM. But it seems that it's not very effective to
                // performance improvement in this case. A_vals[fragment_y_idx]
                // = A_thread_block_tile[A_thread_block_tile_row_idx +
                // fragment_y_idx][A_thread_block_tile_col_idx];
                A_vals[fragment_y_idx] =
                    A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                                  [A_thread_block_tile_row_idx +
                                                   fragment_y_idx];
            }

            for (size_t fragment_y_idx{0U};
                 fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y; ++fragment_y_idx)
            {
                for (size_t fragment_x_idx{0U};
                     fragment_x_idx < BLOCK_FRAGMENT_SIZE_X; ++fragment_x_idx)
                {
                    C_thread_results[fragment_y_idx][fragment_x_idx] +=
                        A_vals[fragment_y_idx] * B_vals[fragment_x_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t fragment_y_idx{0U}; fragment_y_idx < BLOCK_FRAGMENT_SIZE_Y;
         ++fragment_y_idx)
    {
        for (size_t fragment_x_idx{0U}; fragment_x_idx < BLOCK_FRAGMENT_SIZE_X;
             ++fragment_x_idx)
        {
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                threadIdx.x / (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                    BLOCK_FRAGMENT_SIZE_Y +
                fragment_y_idx};
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                threadIdx.x % (BLOCK_TILE_SIZE_X / BLOCK_FRAGMENT_SIZE_X) *
                    BLOCK_FRAGMENT_SIZE_X +
                fragment_x_idx};
            if (C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * n + C_col_idx] =
                    alpha * C_thread_results[fragment_y_idx][fragment_x_idx] +
                    beta * C[C_row_idx * n + C_col_idx];
            }
        }
    }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v7(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    // constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    // Each thread in the block will process BLOCK_FRAGMENT_SIZE_Y *
    // BLOCK_FRAGMENT_SIZE_Y elements. constexpr unsigned int
    // BLOCK_FRAGMENT_SIZE_X{4U}; constexpr unsigned int
    // BLOCK_FRAGMENT_SIZE_Y{4U};
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_X{
        8U}; // This might need to be named as THREAD_FRAGMENT_SIZE_X.
    constexpr unsigned int BLOCK_FRAGMENT_SIZE_Y{8U};
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (BLOCK_FRAGMENT_SIZE_X * BLOCK_FRAGMENT_SIZE_Y)};
    static_assert(BLOCK_TILE_SIZE_X % BLOCK_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % BLOCK_FRAGMENT_SIZE_Y == 0U);
    // These are for DRAM coalesced read/write.
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v7<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            BLOCK_FRAGMENT_SIZE_X, BLOCK_FRAGMENT_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// 2D warp tiling
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <
    typename T, unsigned int BLOCK_TILE_SIZE_X, unsigned int BLOCK_TILE_SIZE_Y,
    unsigned int BLOCK_TILE_SIZE_K, unsigned int WARP_FRAGMENT_SIZE_X,
    unsigned int WARP_FRAGMENT_SIZE_Y, unsigned int THREAD_FRAGMENT_SIZE_X,
    unsigned int THREAD_FRAGMENT_SIZE_Y, unsigned int NUM_THREADS_PER_WARP_X,
    unsigned int NUM_THREADS_PER_WARP_Y,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__global__ void gemm_v8(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_FRAGMENT_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_FRAGMENT_SIZE_Y};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_FRAGMENT_SIZE_X /
        (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_FRAGMENT_SIZE_Y /
        (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_FRAGMENT_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_FRAGMENT_SIZE_Y] = {
        static_cast<T>(0)};

    // size_t const num_threads{blockDim.x};
    constexpr size_t num_threads{NUM_THREADS_PER_BLOCK};

    size_t const warp_linear_idx{threadIdx.x / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{threadIdx.x % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_FRAGMENT_SIZE_Y *
    // THREAD_FRAGMENT_SIZE_X output values. Specifically, these values
    // corresponds to
    //
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_FRAGMENT_SIZE_Y][THREAD_FRAGMENT_SIZE_X] = {
                          static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // Question/TODO: Can this load function be a warp based function?
        // Load data from A on DRAM to A_thread_block_tile on shared memory.
        // #pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
                            num_threads; // Using NUM_THREADS_PER_BLOCK instead
                                         // of num_threads results in larger
                                         // numerical error. Crazy ?!
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                BLOCK_TILE_SIZE_K};
            size_t const A_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                BLOCK_TILE_SIZE_K};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};
            A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                          [A_thread_block_tile_row_idx] =
                                              A[A_row_idx * k + A_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. T const val{(A_row_idx < m && A_col_idx < k) ?
            // A[A_row_idx * k + A_col_idx] : static_cast<T>(0)};
            // A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //                               [A_thread_block_tile_row_idx] =
            //                               val;
        }
        // Load data from B on DRAM to B_thread_block_tile on shared memory.
        // #pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                BLOCK_TILE_SIZE_X};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                BLOCK_TILE_SIZE_X};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};
            B_thread_block_tile[B_thread_block_tile_row_idx]
                               [B_thread_block_tile_col_idx] =
                                   B[B_row_idx * n + B_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. T const val{(B_row_idx < k && B_col_idx < n) ?
            // B[B_row_idx * n + B_col_idx] : static_cast<T>(0)};
            // B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx] = val;
        }
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
// Can use pragma unroll to unroll these static loops to see if there is a
// performance gain.
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                    thread_tile_row_idx *
                        (WARP_FRAGMENT_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
#pragma unroll
                for (size_t thread_fragment_y_idx{0U};
                     thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                     ++thread_fragment_y_idx)
                {
                    A_vals[thread_tile_row_idx][thread_fragment_y_idx] =
                        A_thread_block_tile_transposed
                            [A_thread_block_tile_col_idx]
                            [A_thread_block_tile_row_idx +
                             thread_fragment_y_idx];
                }
            }
#pragma unroll
            for (size_t thread_tile_col_idx{0U};
                 thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_FRAGMENT_SIZE_X +
                    thread_tile_col_idx *
                        (WARP_FRAGMENT_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X};
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx < THREAD_FRAGMENT_SIZE_X;
                     ++thread_fragment_x_idx)
                {
                    B_vals[thread_tile_col_idx][thread_fragment_x_idx] =
                        B_thread_block_tile[B_thread_block_tile_row_idx]
                                           [B_thread_block_tile_col_idx +
                                            thread_fragment_x_idx];
                }
            }

// Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// products.
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_col_idx)
                {
#pragma unroll
                    for (size_t thread_fragment_y_idx{0U};
                         thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                         ++thread_fragment_y_idx)
                    {
#pragma unroll
                        for (size_t thread_fragment_x_idx{0U};
                             thread_fragment_x_idx < THREAD_FRAGMENT_SIZE_X;
                             ++thread_fragment_x_idx)
                        {
                            C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx]
                                            [thread_fragment_y_idx]
                                            [thread_fragment_x_idx] +=
                                A_vals[thread_tile_row_idx]
                                      [thread_fragment_y_idx] *
                                B_vals[thread_tile_col_idx]
                                      [thread_fragment_x_idx];
                        }
                    }
                }
            }
        }
        // We can use syncwarp now.
        __syncwarp();
    }
    // Need a synchronization before writing the results to DRAM.
    __syncthreads();

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_col_idx{0U};
             thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_col_idx)
        {
#pragma unroll
            for (size_t thread_fragment_y_idx{0U};
                 thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                 ++thread_fragment_y_idx)
            {
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx < THREAD_FRAGMENT_SIZE_X;
                     ++thread_fragment_x_idx)
                {
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                        thread_tile_row_idx * (WARP_FRAGMENT_SIZE_Y /
                                               NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y +
                        thread_fragment_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_FRAGMENT_SIZE_X +
                        thread_tile_col_idx * (WARP_FRAGMENT_SIZE_X /
                                               NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X +
                        thread_fragment_x_idx};
                    if (C_row_idx < m && C_col_idx < n)
                    {
                        C[C_row_idx * n + C_col_idx] =
                            alpha * C_thread_results[thread_tile_row_idx]
                                                    [thread_tile_col_idx]
                                                    [thread_fragment_y_idx]
                                                    [thread_fragment_x_idx] +
                            beta * C[C_row_idx * n + C_col_idx];
                    }
                }
            }
        }
    }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
void launch_gemm_kernel_v8(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    // constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};

    constexpr unsigned int WARP_FRAGMENT_SIZE_X{32U};
    constexpr unsigned int WARP_FRAGMENT_SIZE_Y{64U};

    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X /
                                       WARP_FRAGMENT_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y /
                                       WARP_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_FRAGMENT_SIZE_Y == 0U);

    constexpr unsigned int THREAD_FRAGMENT_SIZE_X{4U};
    constexpr unsigned int THREAD_FRAGMENT_SIZE_Y{4U};

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};

    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{WARP_FRAGMENT_SIZE_X /
    //                                                     (THREAD_FRAGMENT_SIZE_X
    //                                                     * NUM_THREADS_PER_WARP_X)};
    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{WARP_FRAGMENT_SIZE_Y /
    //                                                     (THREAD_FRAGMENT_SIZE_Y
    //                                                     * NUM_THREADS_PER_WARP_Y)};

    static_assert(WARP_FRAGMENT_SIZE_X %
                      (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X) ==
                  0U);
    static_assert(WARP_FRAGMENT_SIZE_Y %
                      (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y) ==
                  0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v8<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            WARP_FRAGMENT_SIZE_X, WARP_FRAGMENT_SIZE_Y, THREAD_FRAGMENT_SIZE_X,
            THREAD_FRAGMENT_SIZE_Y, NUM_THREADS_PER_WARP_X,
            NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// 2D warp tiling with vectorized data IO
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <
    typename T, unsigned int BLOCK_TILE_SIZE_X, unsigned int BLOCK_TILE_SIZE_Y,
    unsigned int BLOCK_TILE_SIZE_K, unsigned int WARP_FRAGMENT_SIZE_X,
    unsigned int WARP_FRAGMENT_SIZE_Y, unsigned int THREAD_FRAGMENT_SIZE_X,
    unsigned int THREAD_FRAGMENT_SIZE_Y, unsigned int NUM_THREADS_PER_WARP_X,
    unsigned int NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v9(size_t m, size_t n, size_t k, float alpha, T const* A,
                        T const* B, float beta, T* C)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_FRAGMENT_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_FRAGMENT_SIZE_Y};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_FRAGMENT_SIZE_X /
        (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_FRAGMENT_SIZE_Y /
        (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};

    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_FRAGMENT_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_FRAGMENT_SIZE_Y] = {
        static_cast<T>(0)};

    // size_t const num_threads{blockDim.x};
    constexpr size_t num_threads{NUM_THREADS_PER_BLOCK};

    size_t const warp_linear_idx{threadIdx.x / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{threadIdx.x % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_FRAGMENT_SIZE_Y *
    // THREAD_FRAGMENT_SIZE_X output values. Specifically, these values
    // corresponds to
    //
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_FRAGMENT_SIZE_Y][THREAD_FRAGMENT_SIZE_X] = {
                          static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
// Question / TODO: Can this load function be a warp based function?
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
                            NUM_VECTOR_UNITS /
                            num_threads; // Using NUM_THREADS_PER_BLOCK instead
                                         // of num_threads results in larger
                                         // numerical error. Crazy ?!
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)};
            size_t const A_thread_block_tile_col_idx{
                ((threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                 (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)) *
                NUM_VECTOR_UNITS};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};
            // A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]
            // = A[A_row_idx * k + A_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. float4 const val{(A_row_idx < m && A_col_idx < k) ?
            // *reinterpret_cast<float4 const*>(&A[A_row_idx * k + A_col_idx]) :
            // float4{0.f, 0.f, 0.f, 0.f}};

            // float4 val{0.f, 0.f, 0.f, 0.f};
            // if ((k - A_col_idx) / NUM_VECTOR_UNITS == 0U && (k - A_col_idx) %
            // NUM_VECTOR_UNITS != 0U)
            // {
            //     size_t const num_remains{(k - A_col_idx) % NUM_VECTOR_UNITS};
            //     for (size_t vector_unit_idx{0U}; vector_unit_idx <
            //     num_remains; ++vector_unit_idx)
            //     {
            //         T const val_single{(A_row_idx < m && A_col_idx +
            //         vector_unit_idx < k) ? A[A_row_idx * k + A_col_idx +
            //         vector_unit_idx] : static_cast<T>(0)};
            //         reinterpret_cast<T*>(&val)[vector_unit_idx] = val_single;
            //     }

            //     // T const val{(A_row_idx < m && A_col_idx < k) ? A[A_row_idx
            //     * k + A_col_idx] : static_cast<T>(0)};
            //     //
            //     A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //     //                             [A_thread_block_tile_row_idx]
            //     = val;
            // }
            // else
            // {
            //     val = (A_row_idx < m && A_col_idx < k) ?
            //     *reinterpret_cast<float4 const*>(&A[A_row_idx * k +
            //     A_col_idx]) : float4{0.f, 0.f, 0.f, 0.f};
            //     // val = float4{0.f, 0.f, 0.f, 0.f};
            //     // val = (*reinterpret_cast<float4 const*>(&A[A_row_idx * k +
            //     A_col_idx]));
            // }

            float4 const val{*reinterpret_cast<float4 const*>(
                &A[A_row_idx * k + A_col_idx])};

#pragma unroll
            for (size_t vector_unit_idx{0U}; vector_unit_idx < NUM_VECTOR_UNITS;
                 ++vector_unit_idx)
            {
                A_thread_block_tile_transposed[A_thread_block_tile_col_idx +
                                               vector_unit_idx]
                                              [A_thread_block_tile_row_idx] =
                                                  reinterpret_cast<T const*>(
                                                      &val)[vector_unit_idx];
            }
        }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X /
                            NUM_VECTOR_UNITS / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS) * NUM_VECTOR_UNITS};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. float4 const val{(B_row_idx < k && B_col_idx < n) ?
            // *reinterpret_cast<float4 const*>(&B[B_row_idx * n + B_col_idx]) :
            // float4{0.f, 0.f, 0.f, 0.f}};
            // *reinterpret_cast<float4*>(&B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx]) = val;

            // float4 val{0.f, 0.f, 0.f, 0.f};
            // if ((n - B_col_idx) / NUM_VECTOR_UNITS == 0U && (n - B_col_idx) %
            // NUM_VECTOR_UNITS != 0U)
            // {
            //     size_t const num_remains{(n - B_col_idx) % NUM_VECTOR_UNITS};
            //     for (size_t vector_unit_idx{0U}; vector_unit_idx <
            //     num_remains; ++vector_unit_idx)
            //     {
            //         T const val_single{(B_row_idx < k && B_col_idx +
            //         vector_unit_idx < n) ? B[B_row_idx * n + B_col_idx +
            //         vector_unit_idx] : static_cast<T>(0)};
            //         reinterpret_cast<T*>(&val)[vector_unit_idx] = val_single;
            //     }
            //     // T const val{(A_row_idx < m && A_col_idx < k) ? A[A_row_idx
            //     * k + A_col_idx] : static_cast<T>(0)};
            //     //
            //     A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //     //                             [A_thread_block_tile_row_idx]
            //     = val;
            // }
            // else
            // {
            //     val = (B_row_idx < k && B_col_idx < n) ?
            //     *reinterpret_cast<float4 const*>(&B[B_row_idx * n +
            //     B_col_idx]) : float4{0.f, 0.f, 0.f, 0.f};
            // }

            float4 const val{*reinterpret_cast<float4 const*>(
                &B[B_row_idx * n + B_col_idx])};
            *reinterpret_cast<float4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) = val;
        }
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
// Can use pragma unroll to unroll these static loops to see if there is a
// performance gain.
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                    thread_tile_row_idx *
                        (WARP_FRAGMENT_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
#pragma unroll
                for (size_t thread_fragment_y_idx{0U};
                     thread_fragment_y_idx <
                     THREAD_FRAGMENT_SIZE_Y / NUM_VECTOR_UNITS;
                     ++thread_fragment_y_idx)
                {
                    *reinterpret_cast<float4*>(
                        &A_vals[thread_tile_row_idx]
                               [thread_fragment_y_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<float4 const*>(
                            &A_thread_block_tile_transposed
                                [A_thread_block_tile_col_idx]
                                [A_thread_block_tile_row_idx +
                                 thread_fragment_y_idx * NUM_VECTOR_UNITS]);
                }
            }
#pragma unroll
            for (size_t thread_tile_col_idx{0U};
                 thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_FRAGMENT_SIZE_X +
                    thread_tile_col_idx *
                        (WARP_FRAGMENT_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X};
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx <
                     THREAD_FRAGMENT_SIZE_X / NUM_VECTOR_UNITS;
                     ++thread_fragment_x_idx)
                {
                    *reinterpret_cast<float4*>(
                        &B_vals[thread_tile_col_idx]
                               [thread_fragment_x_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<float4 const*>(
                            &B_thread_block_tile[B_thread_block_tile_row_idx]
                                                [B_thread_block_tile_col_idx +
                                                 thread_fragment_x_idx *
                                                     NUM_VECTOR_UNITS]);
                }
            }

// Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// products.
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_col_idx)
                {
#pragma unroll
                    for (size_t thread_fragment_y_idx{0U};
                         thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                         ++thread_fragment_y_idx)
                    {
#pragma unroll
                        for (size_t thread_fragment_x_idx{0U};
                             thread_fragment_x_idx < THREAD_FRAGMENT_SIZE_X;
                             ++thread_fragment_x_idx)
                        {
                            C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx]
                                            [thread_fragment_y_idx]
                                            [thread_fragment_x_idx] +=
                                A_vals[thread_tile_row_idx]
                                      [thread_fragment_y_idx] *
                                B_vals[thread_tile_col_idx]
                                      [thread_fragment_x_idx];
                        }
                    }
                }
            }
        }
        // We can use syncwarp now.
        __syncwarp();
    }
    // Need a synchronization before writing the results to DRAM.
    __syncthreads();

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_col_idx{0U};
             thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_col_idx)
        {
#pragma unroll
            for (size_t thread_fragment_y_idx{0U};
                 thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                 ++thread_fragment_y_idx)
            {
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx <
                     THREAD_FRAGMENT_SIZE_X / NUM_VECTOR_UNITS;
                     ++thread_fragment_x_idx)
                {
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                        thread_tile_row_idx * (WARP_FRAGMENT_SIZE_Y /
                                               NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y +
                        thread_fragment_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_FRAGMENT_SIZE_X +
                        thread_tile_col_idx * (WARP_FRAGMENT_SIZE_X /
                                               NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X +
                        thread_fragment_x_idx * NUM_VECTOR_UNITS};

                    float4 C_vals{*reinterpret_cast<float4*>(
                        &C[C_row_idx * n + C_col_idx])};
#pragma unroll
                    for (size_t vector_unit_idx{0U};
                         vector_unit_idx < NUM_VECTOR_UNITS; ++vector_unit_idx)
                    {
                        reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                            alpha * C_thread_results[thread_tile_row_idx]
                                                    [thread_tile_col_idx]
                                                    [thread_fragment_y_idx]
                                                    [thread_fragment_x_idx *
                                                         NUM_VECTOR_UNITS +
                                                     vector_unit_idx] +
                            beta *
                                reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                        // reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                        // C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                        // * NUM_VECTOR_UNITS + vector_unit_idx] +
                        // reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                    }
                    *reinterpret_cast<float4*>(&C[C_row_idx * n + C_col_idx]) =
                        C_vals;

                    // if (C_row_idx < m && C_col_idx < n)
                    // {

                    //     // if ((n - C_col_idx) / NUM_VECTOR_UNITS == 0U && (n
                    //     - C_col_idx) % NUM_VECTOR_UNITS != 0U)
                    //     // {
                    //     //     size_t const num_remains{(n - C_col_idx) %
                    //     NUM_VECTOR_UNITS};
                    //     //     for (size_t vector_unit_idx{0U};
                    //     vector_unit_idx < num_remains; ++vector_unit_idx)
                    //     //     {
                    //     //         C[C_row_idx * n + C_col_idx] = alpha *
                    //     C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                    //     + vector_unit_idx] + beta * C[C_row_idx * n +
                    //     C_col_idx];
                    //     //     }
                    //     // }
                    //     // else
                    //     // {
                    //     //     float4
                    //     C_vals{*reinterpret_cast<float4*>(&C[C_row_idx * n +
                    //     C_col_idx])};
                    //     //     #pragma unroll
                    //     //     for (size_t vector_unit_idx{0U};
                    //     vector_unit_idx < NUM_VECTOR_UNITS;
                    //     ++vector_unit_idx)
                    //     //     {
                    //     // reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                    //     alpha *
                    //     C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                    //     + vector_unit_idx] + beta *
                    //     reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                    //     //     }
                    //     //     *reinterpret_cast<float4*>(&C[C_row_idx * n +
                    //     C_col_idx]) = C_vals;
                    //     // }

                    //     float4 C_vals{*reinterpret_cast<float4*>(&C[C_row_idx
                    //     * n + C_col_idx])}; #pragma unroll for (size_t
                    //     vector_unit_idx{0U}; vector_unit_idx <
                    //     NUM_VECTOR_UNITS; ++vector_unit_idx)
                    //     {
                    //         reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                    //         alpha *
                    //         C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                    //         + vector_unit_idx] + beta *
                    //         reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                    //     }
                    //     *reinterpret_cast<float4*>(&C[C_row_idx * n +
                    //     C_col_idx]) = C_vals;

                    //     // float4 C_vals{*reinterpret_cast<float4
                    //     const*>(&C[C_row_idx * n + C_col_idx])};

                    //     // #pragma unroll
                    //     // for (size_t vector_unit_idx{0U}; vector_unit_idx <
                    //     NUM_VECTOR_UNITS; ++vector_unit_idx)
                    //     // {
                    //     //     reinterpret_cast<T*>(&C_vals)[vector_unit_idx]
                    //     = alpha *
                    //     C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                    //     + vector_unit_idx] + beta * reinterpret_cast<T
                    //     const*>(&C_vals)[vector_unit_idx];
                    //     // }

                    //     // if ((n - C_col_idx) / NUM_VECTOR_UNITS == 0U && (n
                    //     - C_col_idx) % NUM_VECTOR_UNITS != 0U)
                    //     // {
                    //     //     size_t const num_remains{(n - C_col_idx) %
                    //     NUM_VECTOR_UNITS};
                    //     //     for (size_t vector_unit_idx{0U};
                    //     vector_unit_idx < num_remains; ++vector_unit_idx)
                    //     //     {
                    //     //         // T const val_single{(B_row_idx < k &&
                    //     B_col_idx + vector_unit_idx < n) ? B[B_row_idx * n +
                    //     B_col_idx + vector_unit_idx] : static_cast<T>(0)};
                    //     //         //
                    //     reinterpret_cast<T*>(&val)[vector_unit_idx] =
                    //     val_single;

                    //     //         C[C_row_idx * n + C_col_idx +
                    //     vector_unit_idx] = reinterpret_cast<T
                    //     const*>(&C_vals)[vector_unit_idx];
                    //     //     }
                    //     // }
                    //     // else
                    //     // {
                    //     //     *reinterpret_cast<float4*>(&C[C_row_idx * n +
                    //     C_col_idx]) = C_vals;
                    //     // }

                    //     // *reinterpret_cast<float4*>(&C[C_row_idx * n +
                    //     C_col_idx]) = C_vals;

                    // }
                }
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v9(size_t m, size_t n, size_t k, float alpha,
                           T const* A, T const* B, float beta, T* C,
                           cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    // constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int WARP_FRAGMENT_SIZE_X{32U};
    constexpr unsigned int WARP_FRAGMENT_SIZE_Y{64U};

    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X /
                                       WARP_FRAGMENT_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y /
                                       WARP_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_FRAGMENT_SIZE_Y == 0U);

    constexpr unsigned int THREAD_FRAGMENT_SIZE_X{4U};
    constexpr unsigned int THREAD_FRAGMENT_SIZE_Y{4U};
    // 4, 4 slightly better than 8, 8
    // constexpr unsigned int THREAD_FRAGMENT_SIZE_X{8U};
    // constexpr unsigned int THREAD_FRAGMENT_SIZE_Y{8U};

    // To use float4 for 128bit vectorized memory access.
    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};
    static_assert(THREAD_FRAGMENT_SIZE_X % NUM_VECTOR_UNITS == 0);
    static_assert(THREAD_FRAGMENT_SIZE_Y % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0);

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};

    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{WARP_FRAGMENT_SIZE_X /
    //                                                     (THREAD_FRAGMENT_SIZE_X
    //                                                     * NUM_THREADS_PER_WARP_X)};
    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{WARP_FRAGMENT_SIZE_Y /
    //                                                     (THREAD_FRAGMENT_SIZE_Y
    //                                                     * NUM_THREADS_PER_WARP_Y)};

    static_assert(WARP_FRAGMENT_SIZE_X %
                      (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X) ==
                  0U);
    static_assert(WARP_FRAGMENT_SIZE_Y %
                      (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y) ==
                  0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v9<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            WARP_FRAGMENT_SIZE_X, WARP_FRAGMENT_SIZE_Y, THREAD_FRAGMENT_SIZE_X,
            THREAD_FRAGMENT_SIZE_Y, NUM_THREADS_PER_WARP_X,
            NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// Play with some FP16 GEMM
// 2D warp tiling with vectorized data IO
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <
    typename T, unsigned int BLOCK_TILE_SIZE_X, unsigned int BLOCK_TILE_SIZE_Y,
    unsigned int BLOCK_TILE_SIZE_K, unsigned int WARP_FRAGMENT_SIZE_X,
    unsigned int WARP_FRAGMENT_SIZE_Y, unsigned int THREAD_FRAGMENT_SIZE_X,
    unsigned int THREAD_FRAGMENT_SIZE_Y, unsigned int NUM_THREADS_PER_WARP_X,
    unsigned int NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v10(size_t m, size_t n, size_t k, float alpha, T const* A,
                         T const* B, float beta, T* C)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_FRAGMENT_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_FRAGMENT_SIZE_Y};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_FRAGMENT_SIZE_X /
        (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_FRAGMENT_SIZE_Y /
        (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};

    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // B_vals is cached in the register.
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_FRAGMENT_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals is cached in the register.
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_FRAGMENT_SIZE_Y] = {
        static_cast<T>(0)};

    // size_t const num_threads{blockDim.x};
    constexpr size_t num_threads{NUM_THREADS_PER_BLOCK};

    size_t const warp_linear_idx{threadIdx.x / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{threadIdx.x % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_FRAGMENT_SIZE_Y *
    // THREAD_FRAGMENT_SIZE_X output values. Specifically, these values
    // corresponds to
    //
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_FRAGMENT_SIZE_Y][THREAD_FRAGMENT_SIZE_X] = {
                          static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
// Question / TODO: Can this load function be a warp based function?
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
                            NUM_VECTOR_UNITS /
                            num_threads; // Using NUM_THREADS_PER_BLOCK instead
                                         // of num_threads results in larger
                                         // numerical error. Crazy ?!
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)};
            size_t const A_thread_block_tile_col_idx{
                ((threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                 (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)) *
                NUM_VECTOR_UNITS};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};
            // A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx]
            // = A[A_row_idx * k + A_col_idx];
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. float4 const val{(A_row_idx < m && A_col_idx < k) ?
            // *reinterpret_cast<float4 const*>(&A[A_row_idx * k + A_col_idx]) :
            // float4{0.f, 0.f, 0.f, 0.f}};

            // float4 val{0.f, 0.f, 0.f, 0.f};
            // if ((k - A_col_idx) / NUM_VECTOR_UNITS == 0U && (k - A_col_idx) %
            // NUM_VECTOR_UNITS != 0U)
            // {
            //     size_t const num_remains{(k - A_col_idx) % NUM_VECTOR_UNITS};
            //     for (size_t vector_unit_idx{0U}; vector_unit_idx <
            //     num_remains; ++vector_unit_idx)
            //     {
            //         T const val_single{(A_row_idx < m && A_col_idx +
            //         vector_unit_idx < k) ? A[A_row_idx * k + A_col_idx +
            //         vector_unit_idx] : static_cast<T>(0)};
            //         reinterpret_cast<T*>(&val)[vector_unit_idx] = val_single;
            //     }

            //     // T const val{(A_row_idx < m && A_col_idx < k) ? A[A_row_idx
            //     * k + A_col_idx] : static_cast<T>(0)};
            //     //
            //     A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //     //                             [A_thread_block_tile_row_idx]
            //     = val;
            // }
            // else
            // {
            //     val = (A_row_idx < m && A_col_idx < k) ?
            //     *reinterpret_cast<float4 const*>(&A[A_row_idx * k +
            //     A_col_idx]) : float4{0.f, 0.f, 0.f, 0.f};
            //     // val = float4{0.f, 0.f, 0.f, 0.f};
            //     // val = (*reinterpret_cast<float4 const*>(&A[A_row_idx * k +
            //     A_col_idx]));
            // }

            float4 const val{*reinterpret_cast<float4 const*>(
                &A[A_row_idx * k + A_col_idx])};

#pragma unroll
            for (size_t vector_unit_idx{0U}; vector_unit_idx < NUM_VECTOR_UNITS;
                 ++vector_unit_idx)
            {
                A_thread_block_tile_transposed[A_thread_block_tile_col_idx +
                                               vector_unit_idx]
                                              [A_thread_block_tile_row_idx] =
                                                  reinterpret_cast<T const*>(
                                                      &val)[vector_unit_idx];
            }
        }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X /
                            NUM_VECTOR_UNITS / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS) * NUM_VECTOR_UNITS};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};
            // This boundary checking might slow down the kernel to some extent.
            // That's why some specific data formats are beneficial for CUDA
            // kernels. float4 const val{(B_row_idx < k && B_col_idx < n) ?
            // *reinterpret_cast<float4 const*>(&B[B_row_idx * n + B_col_idx]) :
            // float4{0.f, 0.f, 0.f, 0.f}};
            // *reinterpret_cast<float4*>(&B_thread_block_tile[B_thread_block_tile_row_idx]
            //                    [B_thread_block_tile_col_idx]) = val;

            // float4 val{0.f, 0.f, 0.f, 0.f};
            // if ((n - B_col_idx) / NUM_VECTOR_UNITS == 0U && (n - B_col_idx) %
            // NUM_VECTOR_UNITS != 0U)
            // {
            //     size_t const num_remains{(n - B_col_idx) % NUM_VECTOR_UNITS};
            //     for (size_t vector_unit_idx{0U}; vector_unit_idx <
            //     num_remains; ++vector_unit_idx)
            //     {
            //         T const val_single{(B_row_idx < k && B_col_idx +
            //         vector_unit_idx < n) ? B[B_row_idx * n + B_col_idx +
            //         vector_unit_idx] : static_cast<T>(0)};
            //         reinterpret_cast<T*>(&val)[vector_unit_idx] = val_single;
            //     }
            //     // T const val{(A_row_idx < m && A_col_idx < k) ? A[A_row_idx
            //     * k + A_col_idx] : static_cast<T>(0)};
            //     //
            //     A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
            //     //                             [A_thread_block_tile_row_idx]
            //     = val;
            // }
            // else
            // {
            //     val = (B_row_idx < k && B_col_idx < n) ?
            //     *reinterpret_cast<float4 const*>(&B[B_row_idx * n +
            //     B_col_idx]) : float4{0.f, 0.f, 0.f, 0.f};
            // }

            float4 const val{*reinterpret_cast<float4 const*>(
                &B[B_row_idx * n + B_col_idx])};
            *reinterpret_cast<float4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) = val;
        }
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
// Can use pragma unroll to unroll these static loops to see if there is a
// performance gain.
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                    thread_tile_row_idx *
                        (WARP_FRAGMENT_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
#pragma unroll
                for (size_t thread_fragment_y_idx{0U};
                     thread_fragment_y_idx <
                     THREAD_FRAGMENT_SIZE_Y / NUM_VECTOR_UNITS;
                     ++thread_fragment_y_idx)
                {
                    *reinterpret_cast<float4*>(
                        &A_vals[thread_tile_row_idx]
                               [thread_fragment_y_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<float4 const*>(
                            &A_thread_block_tile_transposed
                                [A_thread_block_tile_col_idx]
                                [A_thread_block_tile_row_idx +
                                 thread_fragment_y_idx * NUM_VECTOR_UNITS]);
                }
            }
#pragma unroll
            for (size_t thread_tile_col_idx{0U};
                 thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_FRAGMENT_SIZE_X +
                    thread_tile_col_idx *
                        (WARP_FRAGMENT_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X};
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx <
                     THREAD_FRAGMENT_SIZE_X / NUM_VECTOR_UNITS;
                     ++thread_fragment_x_idx)
                {
                    *reinterpret_cast<float4*>(
                        &B_vals[thread_tile_col_idx]
                               [thread_fragment_x_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<float4 const*>(
                            &B_thread_block_tile[B_thread_block_tile_row_idx]
                                                [B_thread_block_tile_col_idx +
                                                 thread_fragment_x_idx *
                                                     NUM_VECTOR_UNITS]);
                }
            }

// Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// products.
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_col_idx)
                {
#pragma unroll
                    for (size_t thread_fragment_y_idx{0U};
                         thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                         ++thread_fragment_y_idx)
                    {
#pragma unroll
                        for (size_t thread_fragment_x_idx{0U};
                             thread_fragment_x_idx < THREAD_FRAGMENT_SIZE_X;
                             ++thread_fragment_x_idx)
                        {
                            C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx]
                                            [thread_fragment_y_idx]
                                            [thread_fragment_x_idx] +=
                                A_vals[thread_tile_row_idx]
                                      [thread_fragment_y_idx] *
                                B_vals[thread_tile_col_idx]
                                      [thread_fragment_x_idx];
                        }
                    }
                }
            }
        }
        // We can use syncwarp now.
        __syncwarp();
    }
    // Need a synchronization before writing the results to DRAM.
    __syncthreads();

// Write the results to DRAM.
#pragma unroll
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_col_idx{0U};
             thread_tile_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_col_idx)
        {
#pragma unroll
            for (size_t thread_fragment_y_idx{0U};
                 thread_fragment_y_idx < THREAD_FRAGMENT_SIZE_Y;
                 ++thread_fragment_y_idx)
            {
#pragma unroll
                for (size_t thread_fragment_x_idx{0U};
                     thread_fragment_x_idx <
                     THREAD_FRAGMENT_SIZE_X / NUM_VECTOR_UNITS;
                     ++thread_fragment_x_idx)
                {
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                        thread_tile_row_idx * (WARP_FRAGMENT_SIZE_Y /
                                               NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_FRAGMENT_SIZE_Y +
                        thread_fragment_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_FRAGMENT_SIZE_X +
                        thread_tile_col_idx * (WARP_FRAGMENT_SIZE_X /
                                               NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_FRAGMENT_SIZE_X +
                        thread_fragment_x_idx * NUM_VECTOR_UNITS};

                    float4 C_vals{*reinterpret_cast<float4*>(
                        &C[C_row_idx * n + C_col_idx])};
#pragma unroll
                    for (size_t vector_unit_idx{0U};
                         vector_unit_idx < NUM_VECTOR_UNITS; ++vector_unit_idx)
                    {
                        // reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                        // alpha *
                        // C_thread_results[thread_tile_row_idx][thread_tile_col_idx][thread_fragment_y_idx][thread_fragment_x_idx
                        // * NUM_VECTOR_UNITS + vector_unit_idx] + beta *
                        // reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                        reinterpret_cast<T*>(&C_vals)[vector_unit_idx] =
                            C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx]
                                            [thread_fragment_y_idx]
                                            [thread_fragment_x_idx *
                                                 NUM_VECTOR_UNITS +
                                             vector_unit_idx] +
                            reinterpret_cast<T*>(&C_vals)[vector_unit_idx];
                    }
                    *reinterpret_cast<float4*>(&C[C_row_idx * n + C_col_idx]) =
                        C_vals;
                }
            }
        }
    }
}

template <typename T>
void launch_gemm_kernel_v10(size_t m, size_t n, size_t k, float alpha,
                            T const* A, T const* B, float beta, T* C,
                            cudaStream_t stream)
{
    // This kernel is sensitive to the parameters.
    // How to select good paramters?
    // constexpr unsigned int BLOCK_TILE_SIZE_X{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};
    // constexpr unsigned int BLOCK_TILE_SIZE_K{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

    constexpr unsigned int WARP_FRAGMENT_SIZE_X{32U};
    constexpr unsigned int WARP_FRAGMENT_SIZE_Y{64U};

    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X /
                                       WARP_FRAGMENT_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y /
                                       WARP_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_FRAGMENT_SIZE_Y == 0U);

    // constexpr unsigned int THREAD_FRAGMENT_SIZE_X{4U};
    // constexpr unsigned int THREAD_FRAGMENT_SIZE_Y{4U};
    // 4, 4 slightly better than 8, 8
    constexpr unsigned int THREAD_FRAGMENT_SIZE_X{8U};
    constexpr unsigned int THREAD_FRAGMENT_SIZE_Y{8U};

    // To use float4 for 128bit vectorized memory access.
    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};
    static_assert(THREAD_FRAGMENT_SIZE_X % NUM_VECTOR_UNITS == 0);
    static_assert(THREAD_FRAGMENT_SIZE_Y % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0);

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};

    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{WARP_FRAGMENT_SIZE_X /
    //                                                     (THREAD_FRAGMENT_SIZE_X
    //                                                     * NUM_THREADS_PER_WARP_X)};
    // constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{WARP_FRAGMENT_SIZE_Y /
    //                                                     (THREAD_FRAGMENT_SIZE_Y
    //                                                     * NUM_THREADS_PER_WARP_Y)};

    static_assert(WARP_FRAGMENT_SIZE_X %
                      (THREAD_FRAGMENT_SIZE_X * NUM_THREADS_PER_WARP_X) ==
                  0U);
    static_assert(WARP_FRAGMENT_SIZE_Y %
                      (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y) ==
                  0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v10<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             WARP_FRAGMENT_SIZE_X, WARP_FRAGMENT_SIZE_Y, THREAD_FRAGMENT_SIZE_X,
             THREAD_FRAGMENT_SIZE_Y, NUM_THREADS_PER_WARP_X,
             NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

// Play with some FP16 GEMM TensorCore
// 2D warp tiling with vectorized data IO
// Each thread in the block processes BLOCK_FRAGMENT_SIZE_Y *
// BLOCK_FRAGMENT_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// BLOCK_TILE_SIZE_X / (BLOCK_FRAGMENT_SIZE_Y * BLOCK_FRAGMENT_SIZE_X)
template <
    typename T, unsigned int BLOCK_TILE_SIZE_X, unsigned int BLOCK_TILE_SIZE_Y,
    unsigned int BLOCK_TILE_SIZE_K, unsigned int WARP_FRAGMENT_SIZE_X,
    unsigned int WARP_FRAGMENT_SIZE_Y, unsigned int WMMA_FRAGMENT_SIZE_X,
    unsigned int WMMA_FRAGMENT_SIZE_Y, unsigned int WMMA_FRAGMENT_SIZE_K,
    unsigned int NUM_THREADS_PER_WARP_X, unsigned int NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v11(size_t m, size_t n, size_t k, float alpha, T const* A,
                         T const* B, float beta, T* C)
{
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_FRAGMENT_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_FRAGMENT_SIZE_Y};

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};

    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    constexpr size_t NUM_WMMA_TILES_X{WARP_FRAGMENT_SIZE_X /
                                      WMMA_FRAGMENT_SIZE_X};
    constexpr size_t NUM_WMMA_TILES_Y{WARP_FRAGMENT_SIZE_Y /
                                      WMMA_FRAGMENT_SIZE_Y};
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_FRAGMENT_SIZE_K};

    static_assert(WARP_FRAGMENT_SIZE_X % WMMA_FRAGMENT_SIZE_X == 0U);
    static_assert(WARP_FRAGMENT_SIZE_Y % WMMA_FRAGMENT_SIZE_Y == 0U);
    static_assert(BLOCK_TILE_SIZE_K % WMMA_FRAGMENT_SIZE_K == 0U);

    // Declare the fragments.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_FRAGMENT_SIZE_Y,
                           WMMA_FRAGMENT_SIZE_X, WMMA_FRAGMENT_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_FRAGMENT_SIZE_Y,
                           WMMA_FRAGMENT_SIZE_X, WMMA_FRAGMENT_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_FRAGMENT_SIZE_Y,
                           WMMA_FRAGMENT_SIZE_X, WMMA_FRAGMENT_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_FRAGMENT_SIZE_Y,
                           WMMA_FRAGMENT_SIZE_X, WMMA_FRAGMENT_SIZE_K, T>
        c_frag;

// Make sure the accumulator starts from 0.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    // // B_vals is cached in the register.
    // T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_FRAGMENT_SIZE_X] =
    // {static_cast<T>(0)};
    // // A_vals is cached in the register.
    // T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_FRAGMENT_SIZE_Y] =
    // {static_cast<T>(0)};

    // size_t const num_threads{blockDim.x};
    constexpr size_t num_threads{NUM_THREADS_PER_BLOCK};

    size_t const warp_linear_idx{threadIdx.x / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{threadIdx.x % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // Number of outer loops to perform the sum of inner products.
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1U) /
                                        BLOCK_TILE_SIZE_K};
    // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_FRAGMENT_SIZE_Y *
    // THREAD_FRAGMENT_SIZE_X output values. Specifically, these values
    // corresponds to
    //
    // T
    // C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X][THREAD_FRAGMENT_SIZE_Y][THREAD_FRAGMENT_SIZE_X]
    // = {
    //     static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
// Question / TODO: Can this load function be a warp based function?
// Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K /
                            NUM_VECTOR_UNITS / num_threads;
             ++load_idx)
        {
            size_t const A_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)};
            size_t const A_thread_block_tile_col_idx{
                ((threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                 (BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS)) *
                NUM_VECTOR_UNITS};
            size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                   A_thread_block_tile_row_idx};
            size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   A_thread_block_tile_col_idx};

            float4 const val{*reinterpret_cast<float4 const*>(
                &A[A_row_idx * k + A_col_idx])};

#pragma unroll
            for (size_t vector_unit_idx{0U}; vector_unit_idx < NUM_VECTOR_UNITS;
                 ++vector_unit_idx)
            {
                A_thread_block_tile_transposed[A_thread_block_tile_col_idx +
                                               vector_unit_idx]
                                              [A_thread_block_tile_row_idx] =
                                                  reinterpret_cast<T const*>(
                                                      &val)[vector_unit_idx];
            }
        }
// Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
        for (size_t load_idx{0U};
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X /
                            NUM_VECTOR_UNITS / num_threads;
             ++load_idx)
        {
            size_t const B_thread_block_tile_row_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) /
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS)};
            size_t const B_thread_block_tile_col_idx{
                (threadIdx.x + load_idx * NUM_THREADS_PER_BLOCK) %
                (BLOCK_TILE_SIZE_X / NUM_VECTOR_UNITS) * NUM_VECTOR_UNITS};
            size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                   B_thread_block_tile_row_idx};
            size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                   B_thread_block_tile_col_idx};

            float4 const val{*reinterpret_cast<float4 const*>(
                &B[B_row_idx * n + B_col_idx])};
            *reinterpret_cast<float4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) = val;
        }
        __syncthreads();

// Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// shared memory as A_thread_block_tile and B_thread_block_tile,
// respectively. This inner product is further decomposed to
// BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// both A_thread_block_tile and B_thread_block_tile can be cached in the
// register.
// Can use pragma unroll to unroll these static loops to see if there is a
// performance gain.
#pragma unroll
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
        {
#pragma unroll
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
            {
#pragma unroll
                for (size_t wmma_tile_col_idx{0U};
                     wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
                {
                    // // Load the fragment from shared memory.
                    nvcuda::wmma::load_matrix_sync(
                        a_frags[wmma_tile_row_idx],
                        &A_thread_block_tile_transposed
                            [k_i * WMMA_FRAGMENT_SIZE_K]
                            [warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                             wmma_tile_row_idx * WMMA_FRAGMENT_SIZE_Y],
                        BLOCK_TILE_SIZE_Y);
                    nvcuda::wmma::load_matrix_sync(
                        b_frags[wmma_tile_col_idx],
                        &B_thread_block_tile[k_i * WMMA_FRAGMENT_SIZE_K]
                                            [warp_col_idx *
                                                 WARP_FRAGMENT_SIZE_X +
                                             wmma_tile_col_idx *
                                                 WMMA_FRAGMENT_SIZE_Y],
                        BLOCK_TILE_SIZE_X);

                    // nvcuda::wmma::load_matrix_sync(a_frags[wmma_tile_row_idx],
                    // &A_thread_block_tile_transposed[0][warp_row_idx *
                    // WARP_FRAGMENT_SIZE_Y + wmma_tile_row_idx *
                    // WMMA_FRAGMENT_SIZE_Y], WARP_FRAGMENT_SIZE_Y);
                    // nvcuda::wmma::load_matrix_sync(b_frags[wmma_tile_col_idx],
                    // &B_thread_block_tile[0][0], WARP_FRAGMENT_SIZE_X);

                    // Perform the matrix multiplication.
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                        a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
        // We can use syncwarp now.
        __syncwarp();
    }
    // Need a synchronization before writing the results to DRAM.
    __syncthreads();

// Write the results to DRAM.
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // Load the fragment from shared memory.
            // T* matrix_mma_c_mptr{C + blockIdx.y * BLOCK_TILE_SIZE_Y +
            // warp_row_idx * WARP_FRAGMENT_SIZE_Y + wmma_tile_row_idx *
            // WMMA_FRAGMENT_SIZE_Y};
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                    wmma_tile_row_idx * WMMA_FRAGMENT_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_FRAGMENT_SIZE_X +
                   wmma_tile_col_idx * WMMA_FRAGMENT_SIZE_X],
                n, nvcuda::wmma::mem_row_major);
            // Perform scaling and addition.
            for (uint32_t i = 0; i < c_frag.num_elements; ++i)
            {
                // c_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] = alpha *
                // acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] + beta *
                // c_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i];
                c_frag.x[i] =
                    acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] +
                    c_frag.x[i];
            }
            // Store the fragment back to shared memory.
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_FRAGMENT_SIZE_Y +
                    wmma_tile_row_idx * WMMA_FRAGMENT_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_FRAGMENT_SIZE_X +
                   wmma_tile_col_idx * WMMA_FRAGMENT_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename T>
void launch_gemm_kernel_v11(size_t m, size_t n, size_t k, float alpha,
                            T const* A, T const* B, float beta, T* C,
                            cudaStream_t stream)
{

    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};

    constexpr unsigned int WARP_FRAGMENT_SIZE_X{32U};
    constexpr unsigned int WARP_FRAGMENT_SIZE_Y{64U};

    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X /
                                       WARP_FRAGMENT_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y /
                                       WARP_FRAGMENT_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_FRAGMENT_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_FRAGMENT_SIZE_Y == 0U);

    constexpr unsigned int WMMA_FRAGMENT_SIZE_X{16U};
    constexpr unsigned int WMMA_FRAGMENT_SIZE_Y{16U};
    constexpr unsigned int WMMA_FRAGMENT_SIZE_K{16U};

    // To use float4 for 128bit vectorized memory access.
    constexpr size_t NUM_VECTOR_UNITS{sizeof(float4) / sizeof(T)};
    // static_assert(THREAD_FRAGMENT_SIZE_X % NUM_VECTOR_UNITS == 0);
    // static_assert(THREAD_FRAGMENT_SIZE_Y % NUM_VECTOR_UNITS == 0);
    // static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0);
    // static_assert(BLOCK_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0);
    // static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0);

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};

    // static_assert(WARP_FRAGMENT_SIZE_X % (THREAD_FRAGMENT_SIZE_X *
    // NUM_THREADS_PER_WARP_X) == 0U); static_assert(WARP_FRAGMENT_SIZE_Y %
    // (THREAD_FRAGMENT_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};
    static_assert(NUM_THREADS_PER_BLOCK == 256U);

    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    gemm_v11<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             WARP_FRAGMENT_SIZE_X, WARP_FRAGMENT_SIZE_Y, WMMA_FRAGMENT_SIZE_X,
             WMMA_FRAGMENT_SIZE_Y, WMMA_FRAGMENT_SIZE_K, NUM_THREADS_PER_WARP_X,
             NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, alpha, A, B, beta, C);
    CHECK_LAST_CUDA_ERROR();
}

int main()
{
    constexpr unsigned int num_repeats{10U};
    constexpr unsigned int num_warmups{10U};

    // Query deive name and peak memory bandwidth.
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

    constexpr size_t m{4096U};
    constexpr size_t k{4096U};
    constexpr size_t n{4096U};

    // constexpr size_t m{2048U};
    // constexpr size_t k{2048U};
    // constexpr size_t n{2048U};

    // constexpr size_t m{1024U};
    // constexpr size_t k{1024U};
    // constexpr size_t n{1024U};

    // constexpr size_t m{312U};
    // constexpr size_t k{325U};
    // constexpr size_t n{631U};

    // constexpr size_t m{513U};
    // constexpr size_t k{513U};
    // constexpr size_t n{521U};

    std::cout << "GEMM Sizes: " << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << std::endl;

    float* d_mat_a;
    float* d_mat_b;
    float* d_mat_c;

    float* h_mat_a;
    float* h_mat_b;
    float* h_mat_c;
    float* h_mat_c_cpu;

    float const alpha{1.0f};
    float const beta{0.0f};

    float const abs_err_tol{1.0};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    CHECK_CUDA_ERROR(cudaMallocHost(&h_mat_a, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_mat_b, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_mat_c, m * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_mat_c_cpu, m * n * sizeof(float)));

    init_random(h_mat_a, m * k);
    init_random(h_mat_b, k * n);
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c_cpu, 0, m * n * sizeof(float), stream));

    CHECK_CUDA_ERROR(cudaMallocAsync(&d_mat_a, m * k * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_mat_b, k * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_mat_c, m * n * sizeof(float), stream));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_a, h_mat_a, m * k * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_b, h_mat_b, k * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c_cpu, h_mat_c,
                                     m * n * sizeof(float),
                                     cudaMemcpyHostToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Use CPU to compute GEMM.
    // launch_gemm_cpu(m, n, k, alpha, h_mat_a, h_mat_b, beta, h_mat_c_cpu);

    cublasHandle_t handle;
    CHECK_CUBLASS_ERROR(cublasCreate(&handle));
    CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

    // Use cuBLAS to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_cublas(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c, handle);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    memcpy(h_mat_c_cpu, h_mat_c, m * n * sizeof(float));
    // Check the results.
    // assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_cublas{
        [m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
         handle](cudaStream_t stream) {
            launch_gemm_cublas(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                               handle);
        }};
    float const latency_cublas{
        measure_performance(function_cublas, stream, num_repeats, num_warmups)};
    std::cout << "cuBLAS Latency: " << latency_cublas << " ms" << std::endl;
    std::cout << "cuBLAS Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_cublas * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "cuBLAS TFLOPS: "
              << (2.0 * m * k * n) / (latency_cublas * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v0 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v0(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v0{
        std::bind(launch_gemm_kernel_v0<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v0{measure_performance(
        function_kernel_v0, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V0 Latency: " << latency_kernel_v0 << " ms"
              << std::endl;
    std::cout << "Kernel V0 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v0 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V0 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v0 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v1 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v1(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v1{
        std::bind(launch_gemm_kernel_v1<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v1{measure_performance(
        function_kernel_v1, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V1 Latency: " << latency_kernel_v1 << " ms"
              << std::endl;
    std::cout << "Kernel V1 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v1 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V1 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v1 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v2 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v2(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Check the results.
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v2{
        std::bind(launch_gemm_kernel_v2<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v2{measure_performance(
        function_kernel_v2, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V2 Latency: " << latency_kernel_v2 << " ms"
              << std::endl;
    std::cout << "Kernel V2 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v2 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V2 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v2 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v5 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v5(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v5{
        std::bind(launch_gemm_kernel_v5<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v5{measure_performance(
        function_kernel_v5, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V5 Latency: " << latency_kernel_v5 << " ms"
              << std::endl;
    std::cout << "Kernel V5 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v5 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V5 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v5 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v5 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_siboehm_v5(m, n, k, alpha, d_mat_a, d_mat_b, beta,
                                  d_mat_c, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_siboehm_v5{
        std::bind(launch_gemm_kernel_siboehm_v5<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_siboehm_v5{measure_performance(
        function_kernel_siboehm_v5, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V5 Siboehm Latency: " << latency_kernel_siboehm_v5
              << " ms" << std::endl;
    std::cout << "Kernel V5 Siboehm Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_siboehm_v5 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V5 Siboehm TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_siboehm_v5 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v6 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v6(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v6{
        std::bind(launch_gemm_kernel_v6<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v6{measure_performance(
        function_kernel_v6, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V6 Latency: " << latency_kernel_v6 << " ms"
              << std::endl;
    std::cout << "Kernel V6 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v6 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V6 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v6 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v7 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v7(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v7{
        std::bind(launch_gemm_kernel_v7<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v7{measure_performance(
        function_kernel_v7, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V7 Latency: " << latency_kernel_v7 << " ms"
              << std::endl;
    std::cout << "Kernel V7 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v7 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V7 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v7 * 1e-3) / 1e12
              << std::endl;
    // 125 TFLOPS for GA 102 RTX 3090?
    std::cout << std::endl;

    // Use kernel v8 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v8(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v8{
        std::bind(launch_gemm_kernel_v8<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v8{measure_performance(
        function_kernel_v8, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V8 Latency: " << latency_kernel_v8 << " ms"
              << std::endl;
    std::cout << "Kernel V8 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v8 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V8 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v8 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    // Use kernel v9 to compute GEMM.
    // Set h_mat_c and h_mat_c_cpu to zero for GEMM.
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(h_mat_c, 0, m * n * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_c, h_mat_c, m * n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

    launch_gemm_kernel_v9(m, n, k, alpha, d_mat_a, d_mat_b, beta, d_mat_c,
                          stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_mat_c, d_mat_c, m * n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // check_diff(h_mat_c, h_mat_c_cpu, m * n);
    // Check the results.
    assert(all_close(h_mat_c, h_mat_c_cpu, m * n, abs_err_tol));
    // Measure the effective bandwidth.
    std::function<void(cudaStream_t)> function_kernel_v9{
        std::bind(launch_gemm_kernel_v9<float>, m, n, k, alpha, d_mat_a,
                  d_mat_b, beta, d_mat_c, std::placeholders::_1)};
    float const latency_kernel_v9{measure_performance(
        function_kernel_v9, stream, num_repeats, num_warmups)};
    std::cout << "Kernel V9 Latency: " << latency_kernel_v9 << " ms"
              << std::endl;
    std::cout << "Kernel V9 Effective Bandwidth: "
              << ((m * k + k * n + m * n) * sizeof(float)) /
                     (latency_kernel_v9 * 1e-3) / 1e9
              << " GB/s" << std::endl;
    // Compute the TFLOPS.
    std::cout << "Kernel V9 TFLOPS: "
              << (2.0 * m * k * n) / (latency_kernel_v9 * 1e-3) / 1e12
              << std::endl;
    std::cout << std::endl;

    __half* d_mat_a_fp16;
    __half* d_mat_b_fp16;
    __half* d_mat_c_fp16;

    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_a_fp16, m * k * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_b_fp16, k * n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_c_fp16, m * n * sizeof(__half)));

    // // Measure the effective bandwidth.
    // std::function<void(cudaStream_t)> function_kernel_v10_fp16{
    //     std::bind(launch_gemm_kernel_v10<__half>, m, n, k,
    //     __float2half(alpha),
    //               d_mat_a_fp16, d_mat_b_fp16, __float2half(beta),
    //               d_mat_c_fp16, std::placeholders::_1)};
    // float const latency_kernel_v10_fp16{measure_performance(
    //     function_kernel_v10_fp16, stream, num_repeats, num_warmups)};
    // std::cout << "Kernel V10 FP16 Latency: " << latency_kernel_v10_fp16 << "
    // ms"
    //           << std::endl;
    // std::cout << "Kernel V10 FP16 Effective Bandwidth: "
    //           << ((m * k + k * n + m * n) * sizeof(__half)) /
    //                  (latency_kernel_v10_fp16 * 1e-3) / 1e9
    //           << " GB/s" << std::endl;
    // // Compute the TFLOPS.
    // std::cout << "Kernel V10 FP16 TFLOPS: "
    //           << (2.0 * m * k * n) / (latency_kernel_v10_fp16 * 1e-3) / 1e12
    //           << std::endl;
    // std::cout << std::endl;

    // // Measure the effective bandwidth.
    // std::function<void(cudaStream_t)> function_kernel_v11_fp16{
    //     std::bind(launch_gemm_kernel_v11<__half>, m, n, k,
    //     __float2half(alpha),
    //               d_mat_a_fp16, d_mat_b_fp16, __float2half(beta),
    //               d_mat_c_fp16, std::placeholders::_1)};
    // float const latency_kernel_v11_fp16{measure_performance(
    //     function_kernel_v11_fp16, stream, num_repeats, num_warmups)};
    // std::cout << "Kernel V11 FP16 Tensor Core Latency: "
    //           << latency_kernel_v11_fp16 << " ms" << std::endl;
    // std::cout << "Kernel V11 FP16 Tensor Core Effective Bandwidth: "
    //           << ((m * k + k * n + m * n) * sizeof(__half)) /
    //                  (latency_kernel_v11_fp16 * 1e-3) / 1e9
    //           << " GB/s" << std::endl;
    // // Compute the TFLOPS.
    // std::cout << "Kernel V11 FP16 Tensor Core TFLOPS: "
    //           << (2.0 * m * k * n) / (latency_kernel_v11_fp16 * 1e-3) / 1e12
    //           << std::endl;
    // std::cout << std::endl;

    CHECK_CUBLASS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    CHECK_CUDA_ERROR(cudaFree(d_mat_a));
    CHECK_CUDA_ERROR(cudaFree(d_mat_b));
    CHECK_CUDA_ERROR(cudaFree(d_mat_c));

    CHECK_CUDA_ERROR(cudaFree(d_mat_a_fp16));
    CHECK_CUDA_ERROR(cudaFree(d_mat_b_fp16));
    CHECK_CUDA_ERROR(cudaFree(d_mat_c_fp16));

    CHECK_CUDA_ERROR(cudaFreeHost(h_mat_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_mat_b));
    CHECK_CUDA_ERROR(cudaFreeHost(h_mat_c));

    return 0;
}