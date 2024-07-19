cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Algorithms VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)
file(GLOB CUDA_GEMM_HEADERS ${CMAKE_SOURCE_DIR}/include/*.hpp ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Add all the source files in the current directory to build the library
add_library(
    cuda_gemm 
    SHARED 
    cuda_gemm_utils.cu 
    00_non_coalesced_global_memory_access.cu
    01_coalesced_global_memory_access.cu
    02_2d_block_tiling.cu
    02_2d_block_tiling_vectorized_memory_access.cu
    03_2d_block_tiling_1d_thread_tiling.cu
    03_2d_block_tiling_1d_thread_tiling_vectorized_memory_access.cu
    04_2d_block_tiling_2d_thread_tiling.cu
    04_2d_block_tiling_2d_thread_tiling_vectorized_memory_access.cu
    05_2d_block_tiling_2d_thread_tiling_matrix_transpose.cu
    05_2d_block_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access.cu
    06_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_vectorized_memory_access_double_buffered.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access.cu
    07_2d_block_tiling_2d_warp_tiling_2d_thread_tiling_matrix_transpose_wmma_vectorized_memory_access_double_buffered.cu
)

# Add the include directory of the library to the include directories of the project
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)
install(TARGETS cuda_gemm DESTINATION lib)
install(FILES ${CUDA_GEMM_HEADERS} DESTINATION include)

add_executable(profile_cuda_gemm_fp32 profile_cuda_gemm_fp32.cu)
target_link_libraries(profile_cuda_gemm_fp32 cuda_gemm CUDA::cublas)
set_target_properties(profile_cuda_gemm_fp32 PROPERTIES CUDA_ARCHITECTURES native)

add_executable(profile_cuda_gemm_fp16 profile_cuda_gemm_fp16.cu)
target_link_libraries(profile_cuda_gemm_fp16 cuda_gemm CUDA::cublas)
set_target_properties(profile_cuda_gemm_fp16 PROPERTIES CUDA_ARCHITECTURES native)