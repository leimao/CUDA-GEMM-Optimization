cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Optimization VERSION 0.0.1 LANGUAGES CXX CUDA)

find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)