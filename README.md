# CUDA GEMM Optimization

## Introduction

## Usages

### Build Docker Images

```bash
$ docker build -f docker/gemm-cuda.Dockerfile --no-cache --tag=gemm-cuda:12.2.2 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt gemm-cuda:12.2.2
```

If we want to profile the CUDA kernels using [NVIDIA Nsight Compute](/blog/Docker-Nsight-Compute/), we need to add additional flags `--cap-add=SYS_ADMIN` and `--security-opt seccomp=unconfined` when we run the Docker container.

### Build CUDA Kernels

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
$ cmake --install build
```

### Run CUDA Kernels

```bash
$ ./build/src/profile_cuda_gemm_fp32
$ ./build/src/profile_cuda_gemm_fp16
```

## Performances

### FP32 GEMM

| Kernel                            | TFLOPS   |                                                                                                Description |
| :-------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------: |
| cuBLAS GEMM Kernel                | 24.5971  |                                                                                      cuBLAS implementation |
| Custom GEMM Kernel V00            | 0.278129 |                                                                         Non-coalesced global memory access |
| Custom GEMM Kernel V01            | 1.7218   |                                                                             Coalesced global memory access |
| Custom GEMM Kernel V02            | 2.66157  |                                                                                            2D block tiling |
| Custom GEMM Kernel V02 Vectorized | 1.90514  |                                                              2D block tiling with vectorized memory access |
| Custom GEMM Kernel V03            | 8.91318  |                                                                       2D block tiling and 1D thread tiling |
| Custom GEMM Kernel V03 Vectorized | 4.04796  |                                         2D block tiling and 1D thread tiling with vectorized memory access |
| Custom GEMM Kernel V04            | 13.0247  |                                                                       2D block tiling and 2D thread tiling |
| Custom GEMM Kernel V04 Vectorized | 15.027   |                                         2D block tiling and 2D thread tiling with vectorized memory access |
| Custom GEMM Kernel V05            | 11.1448  |                                                  2D block tiling and 2D thread tiling and matrix transpose |
| Custom GEMM Kernel V05 Vectorized | 19.6688  |                    2D block tiling and 2D thread tiling and matrix transpose with vectorized memory access |
| Custom GEMM Kernel V06            | 11.0703  |                               2D block tiling and 2D warp tiling and 2D thread tiling and matrix transpose |
| Custom GEMM Kernel V06 Vectorized | 20.1649  | 2D block tiling and 2D warp tiling and 2D thread tiling and matrix transpose with vectorized memory access |

### FP16 GEMM

| Kernel                            | TFLOPS   |                                                                                                Description |
| :-------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------: |
| cuBLAS GEMM Kernel                | 138.955  |                                                                                      cuBLAS implementation |
| Custom GEMM Kernel V00            | 0.284095 |                                                                         Non-coalesced global memory access |
| Custom GEMM Kernel V01            | 1.7316   |                                                                             Coalesced global memory access |
| Custom GEMM Kernel V02            | 2.46677  |                                                                                       2D block tiling GEMM |
| Custom GEMM Kernel V02 Vectorized | 1.93088  |                                                              2D block tiling with vectorized memory access |
| Custom GEMM Kernel V03            | 8.67563  |                                                                  2D block tiling and 1D thread tiling GEMM |
| Custom GEMM Kernel V03 Vectorized | 2.14047  |                                         2D block tiling and 1D thread tiling with vectorized memory access |
| Custom GEMM Kernel V04            | 20.2746  |                                                                  2D block tiling and 2D thread tiling GEMM |
| Custom GEMM Kernel V04 Vectorized | 22.9001  |                                         2D block tiling and 2D thread tiling with vectorized memory access |
| Custom GEMM Kernel V05            | 18.3736  |                                             2D block tiling and 2D thread tiling and matrix transpose GEMM |
| Custom GEMM Kernel V05 Vectorized | 27.962   |                    2D block tiling and 2D thread tiling and matrix transpose with vectorized memory access |
| Custom GEMM Kernel V06            | 14.7622  |                          2D block tiling and 2D warp tiling and 2D thread tiling and matrix transpose GEMM |
| Custom GEMM Kernel V06 Vectorized | 28.4588  | 2D block tiling and 2D warp tiling and 2D thread tiling and matrix transpose with vectorized memory access |
| Custom GEMM Kernel V07            | 33.808   |                                           2D block tiling and 2D warp tiling and WMMA and matrix transpose |
| Custom GEMM Kernel V07 Vectorized | 46.7866  |             2D block tiling and 2D warp tiling and WMMA and matrix transpose and vectorized memory access. |

Custom GEMM Kernel V00
cuBLAS GEMM Kernel Performance
Latency: 0.989088 ms
Effective Bandwidth: 203.548 GB/s
Effective TFLOPS: 138.955 TFLOPS
Custom GEMM Kernel Performance
Latency: 483.778 ms
Effective Bandwidth: 0.416155 GB/s
Effective TFLOPS: 0.284095 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 0.204451%

Custom GEMM Kernel V01
cuBLAS GEMM Kernel Performance
Latency: 1.13939 ms
Effective Bandwidth: 176.697 GB/s
Effective TFLOPS: 120.625 TFLOPS
Custom GEMM Kernel Performance
Latency: 79.3713 ms
Effective Bandwidth: 2.53652 GB/s
Effective TFLOPS: 1.7316 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.43552%

Custom GEMM Kernel V02
cuBLAS GEMM Kernel Performance
Latency: 1.13968 ms
Effective Bandwidth: 176.652 GB/s
Effective TFLOPS: 120.594 TFLOPS
Custom GEMM Kernel Performance
Latency: 55.7161 ms
Effective Bandwidth: 3.61344 GB/s
Effective TFLOPS: 2.46677 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 2.04551%

Custom GEMM Kernel V02 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 0.990208 ms
Effective Bandwidth: 203.317 GB/s
Effective TFLOPS: 138.798 TFLOPS
Custom GEMM Kernel Performance
Latency: 71.1793 ms
Effective Bandwidth: 2.82844 GB/s
Effective TFLOPS: 1.93088 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.39115%

Custom GEMM Kernel V03
cuBLAS GEMM Kernel Performance
Latency: 1.13869 ms
Effective Bandwidth: 176.806 GB/s
Effective TFLOPS: 120.699 TFLOPS
Custom GEMM Kernel Performance
Latency: 15.842 ms
Effective Bandwidth: 12.7084 GB/s
Effective TFLOPS: 8.67563 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 7.1878%

Custom GEMM Kernel V03 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 0.98992 ms
Effective Bandwidth: 203.377 GB/s
Effective TFLOPS: 138.838 TFLOPS
Custom GEMM Kernel Performance
Latency: 64.2098 ms
Effective Bandwidth: 3.13545 GB/s
Effective TFLOPS: 2.14047 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 1.5417%

Custom GEMM Kernel V04
cuBLAS GEMM Kernel Performance
Latency: 1.13152 ms
Effective Bandwidth: 177.926 GB/s
Effective TFLOPS: 121.464 TFLOPS
Custom GEMM Kernel Performance
Latency: 6.77888 ms
Effective Bandwidth: 29.6991 GB/s
Effective TFLOPS: 20.2746 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 16.6918%

Custom GEMM Kernel V04 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 1.14042 ms
Effective Bandwidth: 176.538 GB/s
Effective TFLOPS: 120.517 TFLOPS
Custom GEMM Kernel Performance
Latency: 6.00166 ms
Effective Bandwidth: 33.5451 GB/s
Effective TFLOPS: 22.9001 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 19.0017%

Custom GEMM Kernel V05
cuBLAS GEMM Kernel Performance
Latency: 1.13971 ms
Effective Bandwidth: 176.647 GB/s
Effective TFLOPS: 120.591 TFLOPS
Custom GEMM Kernel Performance
Latency: 7.48022 ms
Effective Bandwidth: 26.9145 GB/s
Effective TFLOPS: 18.3736 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 15.2363%

Custom GEMM Kernel V05 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 1.13965 ms
Effective Bandwidth: 176.657 GB/s
Effective TFLOPS: 120.598 TFLOPS
Custom GEMM Kernel Performance
Latency: 4.9152 ms
Effective Bandwidth: 40.96 GB/s
Effective TFLOPS: 27.962 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 23.1862%

Custom GEMM Kernel V06
cuBLAS GEMM Kernel Performance
Latency: 1.13955 ms
Effective Bandwidth: 176.672 GB/s
Effective TFLOPS: 120.608 TFLOPS
Custom GEMM Kernel Performance
Latency: 9.31021 ms
Effective Bandwidth: 21.6243 GB/s
Effective TFLOPS: 14.7622 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 12.2398%

Custom GEMM Kernel V06 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 1.14054 ms
Effective Bandwidth: 176.518 GB/s
Effective TFLOPS: 120.503 TFLOPS
Custom GEMM Kernel Performance
Latency: 4.82941 ms
Effective Bandwidth: 41.6876 GB/s
Effective TFLOPS: 28.4588 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 23.6166%

Custom GEMM Kernel V07
cuBLAS GEMM Kernel Performance
Latency: 1.14157 ms
Effective Bandwidth: 176.36 GB/s
Effective TFLOPS: 120.395 TFLOPS
Custom GEMM Kernel Performance
Latency: 4.06528 ms
Effective Bandwidth: 49.5234 GB/s
Effective TFLOPS: 33.808 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 28.0809%

Custom GEMM Kernel V07 Vectorized
cuBLAS GEMM Kernel Performance
Latency: 1.1407 ms
Effective Bandwidth: 176.493 GB/s
Effective TFLOPS: 120.486 TFLOPS
Custom GEMM Kernel Performance
Latency: 2.93757 ms
Effective Bandwidth: 68.5351 GB/s
Effective TFLOPS: 46.7866 TFLOPS
Custom GEMM VS cuBLAS GEMM Performance: 38.8316%
