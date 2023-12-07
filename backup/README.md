nvcc gemm.cu -o gemm -lcublas --gpu-architecture=compute_86 --gpu-code=sm_86

If not considering corner cases, achieving 90% performance of the cuBLAS implementation is easy.

However, if we need to consider corner cases, we may only achieve 70% performance of the cuBLAS implementation.

But this comparison might not still be fair, because for different GEMM shapes, cuBLAS might use different parameters and even implementations/algorithms.
