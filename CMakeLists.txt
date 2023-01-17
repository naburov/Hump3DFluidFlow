cmake_minimum_required(VERSION 3.23)
project(Hump3DWrapModeling LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
include_directories(/usr/local/cuda-11/include)
include_directories(Hump3DWrapModeling)


add_executable(Hump3DFluidFlow
        Hump3DWrapModeling/cell_calculating_functions.cuh
        Hump3DWrapModeling/main.cu
        Hump3DWrapModeling/cell_calculating_functions.cu
        Hump3DWrapModeling/3D_stencil.cuh
        Hump3DWrapModeling/test_kernels.cu
        Hump3DWrapModeling/test_kernels.cuh)

set_target_properties(Hump3DFluidFlow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

