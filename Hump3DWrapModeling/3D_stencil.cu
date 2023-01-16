//
// Created by Буров  Никита on 15.01.2023.
//

#ifndef HUMP3DFLUIDFLOW_3D_STENCIL_H
#define HUMP3DFLUIDFLOW_3D_STENCIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

struct Stencil3D {
    double4 x_less, x_more, y_less, y_more, z_less, z_more, center;

    __device__ double dx_central() {
        return (x_more.w - x_less.w) / (x_more.x - x_less.x);
    }
};

#endif //HUMP3DFLUIDFLOW_3D_STENCIL_H

