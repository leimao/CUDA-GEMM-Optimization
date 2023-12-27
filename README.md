# CUDA GEMM Optimization

$ docker build -f docker/gemm-cuda.Dockerfile --no-cache --tag=gemm-cuda:12.2.2 .
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt gemm-cuda:12.2.2

For profiling, we need additional flags `--cap-add=SYS_ADMIN --security-opt seccomp=unconfined `

cmake -B build

cmake --build build --config Release --parallel

cmake --install build

ncu --set full -f -o profile_cuda_gemm profile_cuda_gemm

TODO

For FP16 GEMM kernels, the accumulation precision is currently using FP16 which results in a large discrepancy from the cuBLAS GEMM kernels which use FP32 for accumulation. Use FP32 for accumulation and then cast to FP16 at the end.

compute-sanitizer --tool memcheck build/src/profile_cuda_gemm

## Performance on RTX 3090

Device Name: NVIDIA GeForce RTX 3090
Memory Size: 23.6694 GB
Peak Bandwitdh: 936.096 GB/s

Matrix Size: M = 4096 N = 4096 K = 4096
Matrix A: 4096 x 4096 Leading Dimension Size = 4096
Matrix B: 4096 x 4096 Leading Dimension Size = 4096
Matrix C: 4096 x 4096 Leading Dimension Size = 4096

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 5.79728 ms
Effective Bandwidth: 34.7277 GB/s
Effective TFLOPS: 23.7075 TFLOPS
Custom GEMM Kernel Performance
Latency: 450.354 ms
Effective Bandwidth: 0.447041 GB/s
Effective TFLOPS: 0.30518 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.28727%

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 5.48152 ms
Effective Bandwidth: 36.7282 GB/s
Effective TFLOPS: 25.0731 TFLOPS
Custom GEMM Kernel Performance
Latency: 77.9926 ms
Effective Bandwidth: 2.58135 GB/s
Effective TFLOPS: 1.7622 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 7.02826%

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 5.68612 ms
Effective Bandwidth: 35.4067 GB/s
Effective TFLOPS: 24.171 TFLOPS
Custom GEMM Kernel Performance
Latency: 50.22 ms
Effective Bandwidth: 4.00889 GB/s
Effective TFLOPS: 2.73674 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 11.3224%

Custom GEMM Kernel V02 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 5.84642 ms
Effective Bandwidth: 34.4359 GB/s
Effective TFLOPS: 23.5082 TFLOPS
Custom GEMM Kernel Performance
Latency: 71.5583 ms
Effective Bandwidth: 2.81346 GB/s
Effective TFLOPS: 1.92066 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 8.17015%

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 5.83081 ms
Effective Bandwidth: 34.5281 GB/s
Effective TFLOPS: 23.5712 TFLOPS
Custom GEMM Kernel Performance
Latency: 17.024 ms
Effective Bandwidth: 11.8261 GB/s
Effective TFLOPS: 8.07326 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 34.2506%

Custom GEMM Kernel V03 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 5.56675 ms
Effective Bandwidth: 36.1659 GB/s
Effective TFLOPS: 24.6892 TFLOPS
Custom GEMM Kernel Performance
Latency: 31.7553 ms
Effective Bandwidth: 6.33994 GB/s
Effective TFLOPS: 4.32807 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 17.5302%

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 5.53805 ms
Effective Bandwidth: 36.3533 GB/s
Effective TFLOPS: 24.8172 TFLOPS
Custom GEMM Kernel Performance
Latency: 9.38423 ms
Effective Bandwidth: 21.4537 GB/s
Effective TFLOPS: 14.6457 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 59.0144%

Custom GEMM Kernel V04 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 5.63644 ms
Effective Bandwidth: 35.7187 GB/s
Effective TFLOPS: 24.384 TFLOPS
Custom GEMM Kernel Performance
Latency: 9.9268 ms
Effective Bandwidth: 20.2811 GB/s
Effective TFLOPS: 13.8452 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 56.78%

Custom GEMM Kernel V05
cuBLAS GEMM Kernel Performance
Latency: 5.78575 ms
Effective Bandwidth: 34.797 GB/s
Effective TFLOPS: 23.7547 TFLOPS
Custom GEMM Kernel Performance
Latency: 9.89891 ms
Effective Bandwidth: 20.3383 GB/s
Effective TFLOPS: 13.8843 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 58.4484%

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 5.91288 ms
Effective Bandwidth: 34.0488 GB/s
Effective TFLOPS: 23.244 TFLOPS
Custom GEMM Kernel Performance
Latency: 7.06867 ms
Effective Bandwidth: 28.4815 GB/s
Effective TFLOPS: 19.4434 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 83.6492%

Custom GEMM Kernel V06
cuBLAS GEMM Kernel Performance
Latency: 5.58623 ms
Effective Bandwidth: 36.0398 GB/s
Effective TFLOPS: 24.6032 TFLOPS
Custom GEMM Kernel Performance
Latency: 9.63915 ms
Effective Bandwidth: 20.8863 GB/s
Effective TFLOPS: 14.2584 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 57.9535%

Custom GEMM Kernel V06 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 5.84207 ms
Effective Bandwidth: 34.4615 GB/s
Effective TFLOPS: 23.5257 TFLOPS
Custom GEMM Kernel Performance
Latency: 7.11953 ms
Effective Bandwidth: 28.2781 GB/s
Effective TFLOPS: 19.3045 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 82.057%
