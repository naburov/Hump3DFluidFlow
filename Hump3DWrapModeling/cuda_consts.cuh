//
// Created by Буров  Никита on 21.01.2023.
//
#include <cuda.h>

#ifndef HUMP3DWRAPMODELING_CUDA_CONSTS_CUH
#define HUMP3DWRAPMODELING_CUDA_CONSTS_CUH
static __constant__  __device__ double x0       = 1.0;
static __constant__  __device__ double z0       = 1.0;
static __constant__  __device__ double eps      = 0.0001;
static __constant__  __device__ double f_second = 0.33;
#endif //HUMP3DWRAPMODELING_CUDA_CONSTS_CUH
