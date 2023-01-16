//
// Created by Буров  Никита on 15.01.2023.
//

#ifndef HUMP3DFLUIDFLOW_3D_STENCIL_H
#define HUMP3DFLUIDFLOW_3D_STENCIL_H

#include "SimulationParams.h"
#include "cell_calculating_functions.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

struct Stencil3D {
    // w - value at the point
    // 7 stencil points
    //              [i,j+1,k] - y_more
    //                 |
    //                 |  [i-1,j,k] x_less
    //                 |  /
    //                 | /
    // [i,j,k-1]----[i,j,k]----[i,j,k+1] z_more
    //                /|
    //               / |
    // x_more[i+1,j,k] |
    //                 |
    //              [i,j-1,k] y_less
    // derivative suffix "l" means left derivative approximation p[..., index, ...] - p[..., index-1, ...]
    // derivative suffix "r" means right derivative approximation p[..., index+1, ...] - p[..., index, ...]
    // derivative suffix "c" means central derivative

    double4 x_less{}, x_more{}, y_less{}, y_more{}, z_less{}, z_more{}, center{};

    __device__ double dx_r() const {
        return (x_more.w - center.w) / (x_more.x - center.x);
    }

    __device__ double dx_l() const {
        return (center.w - x_less.w) / (center.x - x_less.x);
    }

    __device__ double dx_c() const {
        return (x_more.w - x_less.w) / (x_more.x - x_less.x);
    }

    __device__ double dy_r() const {
        return (y_more.w - center.w) / (y_more.y - center.y);
    }

    __device__ double dy_l() const {
        return (center.w - y_less.w) / (center.y - y_less.y);
    }

    __device__ double dy_c() const {
        return (y_more.w - y_less.w) / (y_more.y - y_less.y);
    }

    __device__ double dz_r() const {
        return (z_more.w - center.w) / (z_more.y - center.z);
    }

    __device__ double dz_l() const {
        return (center.w - z_less.w) / (center.z - z_less.z);
    }

    __device__ double dz_c() const {
        return (z_more.w - z_less.w) / (z_more.z - z_less.z);
    }

    __device__ Stencil3D(const double *__restrict__ arr, int i, int j, int k, const SimulationParams &params) {
        center.x = i * params.deltas[0] + params.mins[0];
        center.y = j * params.deltas[1] + params.mins[1];
        center.z = k * params.deltas[2] + params.mins[2];
        center.w = arr[indexof(i, j, k, params)];

        x_more.x = (i + 1) * params.deltas[0] + params.mins[0];
        x_more.y = j * params.deltas[1] + params.mins[1];
        x_more.z = k * params.deltas[2] + params.mins[2];
        x_more.w = arr[indexof((i + 1), j, k, params)];

        x_less.x = (i - 1) * params.deltas[0] + params.mins[0];
        x_less.y = j * params.deltas[1] + params.mins[1];
        x_less.z = k * params.deltas[2] + params.mins[2];
        x_less.w = arr[indexof((i - 1), j, k, params)];

        y_more.x = i * params.deltas[0] + params.mins[0];
        y_more.y = (j + 1) * params.deltas[1] + params.mins[1];
        y_more.z = k * params.deltas[2] + params.mins[2];
        y_more.w = arr[indexof(i, (j + 1), k, params)];

        y_less.x = i * params.deltas[0] + params.mins[0];
        y_less.y = (j - 1) * params.deltas[1] + params.mins[1];
        y_less.z = k * params.deltas[2] + params.mins[2];
        y_less.w = arr[indexof(i, j - 1, k, params)];

        z_more.x = i * params.deltas[0] + params.mins[0];
        z_more.y = j * params.deltas[1] + params.mins[1];
        z_more.z = (k + 1) * params.deltas[2] + params.mins[2];
        z_more.w = arr[indexof(i, j, k + 1, params)];

        z_less.x = i * params.deltas[0] + params.mins[0];
        z_less.y = j * params.deltas[1] + params.mins[1];
        z_less.z = (k - 1) * params.deltas[2] + params.mins[2];
        z_less.w = arr[indexof(i, j, k - 1, params)];
    }
};

#endif //HUMP3DFLUIDFLOW_3D_STENCIL_H

