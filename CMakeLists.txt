cmake_minimum_required(VERSION 3.23)
project(Hump3DFluidFlow)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

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
        Hump3DWrapModeling/calculating_functions.h
        Hump3DWrapModeling/calulating_functions.cpp
        Hump3DWrapModeling/consts.h
        Hump3DWrapModeling/export_functions.cpp
        Hump3DWrapModeling/export_functions.h
        Hump3DWrapModeling/main.cpp Hump3DWrapModeling/Config.cpp Hump3DWrapModeling/Config.h Hump3DWrapModeling/SimulationParams.h)

target_link_libraries(Hump3DFluidFlow PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
        TARGETS Hump3DFluidFlow
        MODULES ${VTK_LIBRARIES}
)

