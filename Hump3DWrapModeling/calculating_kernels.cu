//
// Created by Буров  Никита on 18.01.2023.
//
#include "calculating_kernels.cuh"
#include "cell_calculating_functions.cuh"
#include "SimulationParams.h"
#include<stdio.h>
#include<stdlib.h>

__global__ void h_kernel(double *old_h, double *old_w, double *old_v,
                         double *h, double *w, double *v, double *u,
                         SimulationParams *sim_params) {
    auto threadId     = threadIdx.x + blockIdx.x * blockDim.x;
    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);;
        return;
    }
    if (arr_ids.y == 0) {
        h[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        auto xi1        = sim_params->mins[0];
        auto xi2        = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        auto xi1        = sim_params->mins[0];
        auto xi2        = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = h[linear_index] = c * mu(xi1, xi2, sim_params);

    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);
    }

    Stencil3D h_point(old_h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D w_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D v_point(old_v, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    h[linear_index] = H_point(&h_point, &w_point, &v_point, sim_params);

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        h[bounding_ids] = h[linear_index];
    }
}

__global__ void u_kernel(double *h, double *u,
                         SimulationParams *sim_params) {
    auto threadId     = threadIdx.x + blockIdx.x * blockDim.x;
    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }
    if (arr_ids.y == 0) {
        u[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;

    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
    }

    Stencil3D h_point(h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    u[linear_index] = U_point(&h_point, sim_params);

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        u[bounding_ids] = u[linear_index] + c * sim_params->deltas[1];
    }
}

__global__ void v_func_kernel(double *w, double *v, double *u,
                              SimulationParams *sim_params) {
    auto threadId     = threadIdx.x + blockIdx.x * blockDim.x;
    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        v[linear_index] = 0.0;
        return;
    }
    if (arr_ids.y == 0) {
        v[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        v[linear_index] = 0.0;
    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        v[linear_index] = 0.0;
    }

    Stencil3D w_point(w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D u_point(u, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    v[linear_index] = v_func(&u_point, &w_point, sim_params);
    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        v[bounding_ids]   = 0.0;
    }
}

__global__ void main_kernel(double *old_h, double *old_w, double *old_v, double *old_u,
                            double *h, double *w, double *v, double *u,
                            SimulationParams *sim_params) {
    auto threadId     = threadIdx.x + blockIdx.x * blockDim.x;
    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        v[linear_index] = 0.0;
        w[linear_index] = 0.0;
        return;
    }
    if (arr_ids.y == 0) {
        h[linear_index] = 0.0;
        u[linear_index] = 0.0;
        w[linear_index] = 0.0;
        v[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        auto xi1        = sim_params->mins[0];
        auto xi2        = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);
        w[linear_index] = 0.0;
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        v[linear_index] = 0.0;
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        auto xi1        = sim_params->mins[0];
        auto xi2        = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = h[linear_index] = c * mu(xi1, xi2, sim_params);
        w[linear_index] = 0.0;
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        v[linear_index] = 0.0;

    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = c * mu(xi1, xi2, sim_params);
        w[linear_index] = 0.0;
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        v[linear_index] = 0.0;
    }

    Stencil3D h_point(old_h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D w_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D v_point(old_v, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);

    h[linear_index] = H_point(&h_point, &w_point, &v_point, sim_params);
    __syncthreads();

    Stencil3D new_h_point(h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    u[linear_index] = U_point(&new_h_point, sim_params);
    w[linear_index] = W_point(&new_h_point, &w_point, &v_point, sim_params);
    __syncthreads();

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        h[bounding_ids] = h[linear_index];
        u[bounding_ids] = u[linear_index] + c * sim_params->deltas[1];
        w[bounding_ids] = 0.0;
        v[bounding_ids] = 0.0;
    }

    Stencil3D new_u_point(u, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D new_w_point(w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    v[linear_index] = v_func(&new_u_point, &new_w_point, sim_params);
}

__global__ void integrate_v_kernel(double *v_func, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    auto i        = threadId / sim_params->dims[1];
    auto k        = threadId % sim_params->dims[2];

    // TODO: add bounding conditions

    v_func[indexof(i, 0, k, sim_params)] = 0;
    for (unsigned int j = 1; j < sim_params->dims[1] - 2; j++) {
        v_func[indexof(i, j, k, sim_params)] =
                v_func[indexof(i, j - 1, k, sim_params)]
                - sim_params->deltas[1] * v_func[indexof(i, j, k, sim_params)];
    }
    v_func[indexof(i, sim_params->dims[1] - 1, k, sim_params)] = 0.0;
}

__global__ void reduce_max_kernel(double *array, unsigned int size) {
    extern __shared__ double sdata[];

    unsigned int tid            = threadIdx.x;
    unsigned int i              = blockIdx.x * blockDim.x + tid;
    auto         have_remainder = size & 1;

    if (tid == 0) {
        printf("Hello from thread 0");
    }

    auto index = size >> 1;

    if (have_remainder && tid == 0) {
        sdata[tid] = max(array[i], max(array[i + index], array[size - 1]));
    } else {
        sdata[tid] = max(array[i], array[i + index]);
    }

    for (unsigned int s = index; s > 0; s >>= 1) {
        have_remainder = s & 1;
        if (tid < s) {
            if (have_remainder && tid == 0) {
                sdata[tid] = max(sdata[tid], max(sdata[i + index], sdata[s - 1]));
            } else {
                sdata[tid] = max(sdata[tid], sdata[i + index]);
            }
        }
    }

    if (tid == 0) {
        printf("Max value is: %d", sdata[0]);
    }
}
