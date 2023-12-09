cmake_minimum_required(VERSION 3.28)

project(CUDA-GEMM-Optimization VERSION 0.0.1 LANGUAGES CXX CUDA)

# find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# find_library(CUDA_GEMM_LIBRARIES cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# find_path(CUDA_GEMM_INCLUDE_DIRS cuda_gemm_utils.hpp HINTS ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

# # Executable library files will be installed (compiled and copied) to $CMAKE_INSTALL_PREFIX/lib
# install(TARGETS cuda_gemm 
#         LIBRARY DESTINATION lib 
#         INCLUDES DESTINATION include) 
# # Header files will be copied to $CMAKE_INSTALL_PREFIX/include
# # Do not miss "/"!