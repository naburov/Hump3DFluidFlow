#pragma once

#include <cuda.h>

#include "SimulationParams.h"
#include "3D_stencil.cuh"

//const __device__ double c = 1.0;

__device__ double mu(double xi1, double xi2, SimulationParams *params);

__device__  double c(SimulationParams *params);

__device__ double mu_derivative(double xi1, double xi2, int dim, SimulationParams *params);

__device__ double relaxed_derivative(double a, double derivative_left, double derivative_right);

__device__ double
U_point(Stencil3D *__restrict__ H, SimulationParams *params);

__device__ double
H_point(Stencil3D *__restrict__ U, SimulationParams *params);

__device__ double
H_point(Stencil3D *__restrict__ H, Stencil3D *__restrict__ W, Stencil3D *__restrict__ V, double dp,
        SimulationParams *params);

__device__ double
W_point(Stencil3D *__restrict__ H, Stencil3D *__restrict__ W, Stencil3D *__restrict__ V, SimulationParams *params);

__device__ double v_func(Stencil3D *__restrict__ U, Stencil3D *__restrict__ W, SimulationParams *params);