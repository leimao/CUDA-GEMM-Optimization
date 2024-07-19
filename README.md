# CUDA GEMM Optimization

## Introduction

This repository contains the CUDA kernels for general matrix-matrix multiplication (GEMM) and the corresponding performance analysis. The correctness of the CUDA kernels is guaranteed for any matrix size. The parameters of the CUDA kernels are slightly turned for GEMM 4096 x 4096 x 4096 on an NVIDIA GeForce RTX 3090 GPU. The CUDA kernels should be compatible with any NVIDIA GPUs with compute capability 7.0 or higher.

## Usages

Docker is used to build and run the CUDA kernels. The custom Docker container is built based on the [NVIDIA NGC CUDA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) 12.2.2 Docker container.

Please adjust the base Docker container CUDA version if the host computer has a different CUDA version. Otherwise, weird compilation errors and runtime errors may occur.

### Build Docker Images

To build the custom Docker image, please run the following command.

```bash
$ docker build -f docker/gemm-cuda.Dockerfile --no-cache --tag=gemm-cuda:12.2.2 .
```

### Run Docker Container

To run the custom Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt gemm-cuda:12.2.2
```

If we want to profile the CUDA kernels using [NVIDIA Nsight Compute](https://leimao.github.io/blog/Docker-Nsight-Compute/), we need to add additional flags `--cap-add=SYS_ADMIN` and `--security-opt seccomp=unconfined` when we run the Docker container.

### Build CUDA Kernels

To build the CUDA kernels, please run the following commands inside the Docker container.

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
$ cmake --install build
```

### Run CUDA Kernels

To run the FP32 and FP16 GEMM CUDA kernels, please run the following commands inside the Docker container.

```bash
$ ./build/src/profile_cuda_gemm_fp32
$ ./build/src/profile_cuda_gemm_fp16
```

## Performances

All the experiments are conducted on a single NVIDIA [GeForce RTX 3090 GPU](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf). The performance can vary, sometimes up to 25%, from one measurement to another.

### FP32 GEMM

All the FP32 GEMM kernels cannot utilize the NVIDIA Tensor Cores.

| GEMM Kernel                       | TFLOPS   |                                                                                         Kernel Description |
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

The FP16 custom GEMM kernels V00 to V06 do not utilize the NVIDIA Tensor Cores. The FP16 cuBLAS GEMM kernel and custom GEMM kernels V07 utilize the NVIDIA Tensor Cores.

| GEMM Kernel                       | TFLOPS   |                                                                                         Kernel Description |
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
| Custom GEMM Kernel V07            | 35.2312  |                                           2D block tiling and 2D warp tiling and WMMA and matrix transpose |
| Custom GEMM Kernel V07 Vectorized | 55.0298  |             2D block tiling and 2D warp tiling and WMMA and matrix transpose and vectorized memory access. |

## References

- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
