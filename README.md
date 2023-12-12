# CUDA-GEMM-Optimization

$ docker build -f docker/gemm-cuda.Dockerfile --no-cache --tag=gemm-cuda:0.0.1 .
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt gemm-cuda:0.0.1

cmake -B build

cmake --build build --config Release --parallel

cmake --install build

TODO

For FP16 GEMM kernels, the accumulation precision is currently using FP16 which results in a large discrepancy from the cuBLAS GEMM kernels which use FP32 for accumulation. Use FP32 for accumulation and then cast to FP16 at the end.
