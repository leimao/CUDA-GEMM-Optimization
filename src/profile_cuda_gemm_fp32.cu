#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main()
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    float const fp32_abs_tol{1.0e-3f};
    double const fp32_rel_tol{0.0e-4f};

    constexpr size_t m{4096U};
    constexpr size_t k{4096U};
    constexpr size_t n{4096U};

    constexpr size_t lda{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(n + 16U - 1U) / 16U * 16U};

    static_assert(lda >= k);
    static_assert(ldb >= n);
    static_assert(ldc >= n);

    std::cout << "Matrix Size: " << "M = " << m << " N = " << n << " K = " << k
              << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << ldc << std::endl;
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
             launch_gemm_kernel_v05_vectorized<float>},
            {"Custom GEMM Kernel V06", launch_gemm_kernel_v06<float>},
            {"Custom GEMM Kernel V06 Vectorized",
             launch_gemm_kernel_v06_vectorized<float>},
            {"Custom GEMM Kernel V06 Vectorized Double Buffered",
             launch_gemm_kernel_v06_vectorized_double_buffered<float>},
        };

    for (auto const& gemm_kernel_launch_function : gemm_kernel_launch_functions)
    {
        std::cout << gemm_kernel_launch_function.first << std::endl;
        std::pair<float, float> const gemm_kernel_profile_result{
            profile_gemm<float>(
                m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                fp32_abs_tol, fp32_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return 0;
}