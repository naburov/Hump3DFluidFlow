//
// Created by Буров  Никита on 18.01.2023.
//
#include <cuda.h>
#include "SimulationParams.h"

#ifndef HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH
#define HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH

__global__ void reduce_max_kernel(double *array, unsigned int size);

__global__ void h_kernel(const double *__restrict__ u, double *h, SimulationParams *sim_params);

__global__ void w_init_kernel(double *w, SimulationParams *sim_params);

__global__ void u_init_kernel(double *u, SimulationParams *sim_params);

__global__ void
h_kernel(const double *__restrict__ old_h, const double *__restrict__ old_w, const double *__restrict__ old_v,
         double *h,
         SimulationParams *sim_params);

__global__ void u_kernel(const double *__restrict__ h, double *u, SimulationParams *sim_params);

__global__ void w_kernel(const double *__restrict__ h, const double *__restrict__ old_w, const double *__restrict__ v,
                         double *w, SimulationParams *sim_params);

__global__ void v_func_kernel(const double *__restrict__ w, double *v, const double *__restrict__ u,
                              SimulationParams *sim_params);

__global__ void integrate_v_kernel(double *v_func, SimulationParams *sim_params);

#endif //HUMP3DWRAPMODELING_CALCULATING_KERNELS_CUH
