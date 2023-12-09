# CUDA-GEMM-Optimization

$ docker build -f docker/gemm-cuda.Dockerfile --no-cache --tag=gemm-cuda:0.0.1 .
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt gemm-cuda:0.0.1

cmake -B build

cmake --build build --config Release --parallel

cmake --install build
