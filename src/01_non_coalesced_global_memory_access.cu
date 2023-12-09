#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

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
