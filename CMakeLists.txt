cmake_minimum_required(VERSION 3.23)
project(Hump3DWrapModeling LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
include_directories(/usr/local/cuda-11/include)
include_directories(Hump3DWrapModeling)

find_package(VTK COMPONENTS
        CommonColor
        CommonCore
        InteractionStyle
        CommonDataModel
        IOXML
        FiltersCore
        InfovisCore
        IOCore
        IOExport
        IOGeometry
        IOInfovis
        FiltersGeneral
        )

if (NOT VTK_FOUND)
    message(FATAL_ERROR "Hump3DFluidFlow: Unable to find the VTK build folder.")
endif()

add_executable(Hump3DFluidFlow
        Hump3DWrapModeling/export_functions.h
        Hump3DWrapModeling/export_functions.cpp
        Hump3DWrapModeling/cell_calculating_functions.cuh
        Hump3DWrapModeling/main.cu
        Hump3DWrapModeling/cell_calculating_functions.cu
        Hump3DWrapModeling/3D_stencil.cuh
        Hump3DWrapModeling/test_kernels.cu
        Hump3DWrapModeling/test_kernels.cuh
        Hump3DWrapModeling/calculating_kernels.cu
        Hump3DWrapModeling/calculating_kernels.cuh
        Hump3DWrapModeling/cuda_consts.cuh
        Hump3DWrapModeling/Config.cpp
        Hump3DWrapModeling/Config.h)

set_target_properties(Hump3DFluidFlow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(Hump3DFluidFlow PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
        TARGETS Hump3DFluidFlow
        MODULES ${VTK_LIBRARIES}
)

