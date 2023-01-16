//
// Created by Буров  Никита on 16.01.2023.
//
#include <cuda.h>
#include "SimulationParams.h"
#ifndef HUMP3DWRAPMODELING_TEST_KERNELS_CUH
#define HUMP3DWRAPMODELING_TEST_KERNELS_CUH
__global__ void test_stencil_kernel(double* arr, int i, int j, int k, const SimulationParams& params);
#endif //HUMP3DWRAPMODELING_TEST_KERNELS_CUH
