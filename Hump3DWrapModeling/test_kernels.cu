//
// Created by Буров  Никита on 16.01.2023.
//
#include <cuda.h>
#include "SimulationParams.h"
#include "3D_stencil.cu"

__global__ void test_stencil_kernel(double* arr, int i, int j, int k, const SimulationParams& params){
    auto stencil = Stencil3D(arr, i, j, k, params);
}
