cmake_minimum_required(VERSION 3.23)
project(Hump3DWrapModeling LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

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

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")

add_executable(Hump3DFluidFlow
        Hump3DWrapModeling/cell_calculating_functions.h
        Hump3DWrapModeling/calulating_functions.cpp
        Hump3DWrapModeling/consts.h
        Hump3DWrapModeling/export_functions.cpp
        Hump3DWrapModeling/export_functions.h
        Hump3DWrapModeling/main.cu
        Hump3DWrapModeling/Config.cpp
        Hump3DWrapModeling/Config.h
        Hump3DWrapModeling/SimulationParams.h
        Hump3DWrapModeling/cell_calculating_functions.cu
        Hump3DWrapModeling/3D_stencil.cu)

set_target_properties(Hump3DFluidFlow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(Hump3DFluidFlow PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
        TARGETS Hump3DFluidFlow
        MODULES ${VTK_LIBRARIES}
)

