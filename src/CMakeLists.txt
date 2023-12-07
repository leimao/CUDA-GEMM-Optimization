cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Algorithms VERSION 0.0.1 LANGUAGES CXX CUDA)

# find_package(CUDA REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add all the source files in the current directory to build the library
add_library(gemm SHARED 01_non_coalesced_global_memory_access.cu)
# add_library(gemm SHARED ${CMAKE_CURRENT_SOURCE_DIR})


# Add the include directory of the library to the include directories of the project
target_include_directories(gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# Link the library to the project
target_link_libraries(gemm CUDA::cublas)

# Set the CUDA architecture to compile the code for
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES native)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html