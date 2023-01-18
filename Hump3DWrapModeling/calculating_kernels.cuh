//
// Created by Буров  Никита on 18.01.2023.
//
#include <cuda.h>

#ifndef HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH
#define HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH
__global__ void reduce_max_kernel(double* array, unsigned int size);
__global__ void main_kernel();
#endif //HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH
