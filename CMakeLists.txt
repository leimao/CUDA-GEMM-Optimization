cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Optimization VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
