#pragma once

#include <cuda.h>
#include "SimulationParams.h"

__device__ int indexof(int x, int y, int z, const SimulationParams &params);