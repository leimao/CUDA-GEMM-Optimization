cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Algorithms VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

# Add all the source files in the current directory to build the library
add_library(cuda_gemm SHARED 01_non_coalesced_global_memory_access.cu)
# add_library(gemm SHARED ${CMAKE_CURRENT_SOURCE_DIR})

# Add the include directory of the library to the include directories of the project
target_include_directories(cuda_gemm PUBLIC ${CUDA_GEMM_INCLUDE_DIRS})

# Link the library to the project
target_link_libraries(cuda_gemm CUDA::cublas)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES native)
install(TARGETS cuda_gemm DESTINATION lib)
install(FILES ${CUDA_GEMM_INCLUDE_DIRS}/cuda_gemm.hpp DESTINATION include)
