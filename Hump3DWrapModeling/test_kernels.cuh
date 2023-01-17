//
// Created by Буров  Никита on 16.01.2023.
//
#include <cuda.h>
#include "SimulationParams.h"
#ifndef HUMP3DWRAPMODELING_TEST_KERNELS_CUH
#define HUMP3DWRAPMODELING_TEST_KERNELS_CUH
__global__ void test_stencil_kernel(const double* arr, double* out, int i, int j, int k, SimulationParams* params);
__global__ void test_index_kernel(SimulationParams* params);
__global__ void test_index_back_transform(SimulationParams* params);
#endif //HUMP3DWRAPMODELING_TEST_KERNELS_CUH
