cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Profiling VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# message(STATUS "CUBLAS_LIBRARIES: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
find_library(CUDA_GEMM_LIBRARIES cuda_gemm HINTS ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}/lib)

add_executable(profile_cuda_gemm profile_cuda_gemm.cu)
target_include_directories(profile_cuda_gemm PRIVATE ${CUDA_GEMM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(profile_cuda_gemm PRIVATE ${CUDA_GEMM_LIBRARIES} CUDA::cublas)
set_target_properties(profile_cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)