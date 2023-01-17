#pragma once

#include <cuda.h>
#include "SimulationParams.h"

const double c = 0.33;

// computes flat index given 3D index
__device__ __host__ int indexof(int x, int y, int z, SimulationParams* params);

// computes 3D index from flat
__device__ __host__ int3 indexof(int id, SimulationParams *params);

__device__ double mu(double xi1, double xi2, SimulationParams* params);