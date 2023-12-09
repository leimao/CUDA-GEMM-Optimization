The examples have to be separately installed after the CUDA-GEMM library has been installed.

cmake -B build

cmake --build build --config Release --parallel

cd build

./profile_cuda_gemm
