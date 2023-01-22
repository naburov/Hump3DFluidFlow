//
// Created by Буров  Никита on 18.01.2023.
//
#include "calculating_kernels.cuh"
#include "cuda_consts.cuh"
#include "cell_calculating_functions.cuh"
#include "SimulationParams.h"
#include<stdio.h>
#include<stdlib.h>

__global__ void w_init_kernel(double *w, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;
    auto id    = indexof(threadId, sim_params);
    auto xi1   = id.x * sim_params->deltas[0] + sim_params->mins[0];
    auto theta = id.y * sim_params->deltas[1] + sim_params->mins[1];
    auto xi2   = id.z * sim_params->deltas[2] + sim_params->mins[2];

    if (id.y == 0) {
        w[threadId] = 0;
        return;
    }

    if (xi2 < 0.0) {
        w[threadId] = -mu(xi1, xi2, sim_params) * theta / (1 * theta * theta + 0.00001);
    } else {
        w[threadId] = mu(xi1, xi2, sim_params) * theta / (1 * theta * theta + 0.00001);
    }
}


__global__ void u_init_kernel(double *u, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;

    auto id    = indexof(threadId, sim_params);
    auto xi1   = id.x * sim_params->deltas[0] + sim_params->mins[0];
    auto theta = id.y * sim_params->deltas[1] + sim_params->mins[1];
    auto xi2   = id.z * sim_params->deltas[2] + sim_params->mins[2];

    if (theta < 5.0) {
        u[threadId] = f_second * theta * (1 + 0.2 * mu(xi1, xi2, sim_params));
    } else {
        u[threadId] = f_second * (theta + mu(xi1, xi2, sim_params));
    }
}

__global__ void h_kernel(const double *__restrict__ u, double *h, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;

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
        return;

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
        return;
    }

    Stencil3D u_point(u, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    h[linear_index] = H_point(&u_point, sim_params);

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        h[bounding_ids] = h[linear_index];
    }
}

__global__ void
h_kernel(const double *__restrict__ old_h, const double *__restrict__ old_w, const double *__restrict__ old_v,
         double *h,
         SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;

    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] =
                c * mu(xi1, xi2, sim_params);;
        return;
    }
    if (arr_ids.y == 0) {
        h[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] =
                c * mu(xi1, xi2, sim_params);
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] = h[linear_index] =
                c * mu(xi1, xi2, sim_params);
        return;

    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        auto xi1 = sim_params->mins[0];
        auto xi2 = sim_params->mins[2] + arr_ids.z * sim_params->deltas[2];
        h[linear_index] =
                c * mu(xi1, xi2, sim_params);
        return;
    }

    Stencil3D h_point(old_h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D w_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D v_point(old_v, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);

    // looks like a mistake here
    auto dp = old_v[indexof(arr_ids.x, sim_params->dims[1] - 1, arr_ids.z, sim_params)];
    h[linear_index] =
            H_point(&h_point, &w_point, &v_point, dp, sim_params
            );

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        h[bounding_ids] = h[linear_index];
    }
}

__global__ void u_kernel(const double *__restrict__ h, double *u, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;
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
        return;

    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        // dH/d\theta|_{\theta -> \infty} = 0; -> computed after a stencils computations (u as well)
        // fill in bounding conditions on the dims[1] - 2
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        u[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }

    Stencil3D h_point(h, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    u[linear_index] = U_point(&h_point, sim_params);

    if (arr_ids.y == sim_params->dims[1] - 2) {
        auto bounding_ids = indexof(arr_ids.x, arr_ids.y + 1, arr_ids.z, sim_params);
        u[bounding_ids] = u[linear_index] + c * sim_params->deltas[1];
    }
}

__global__ void v_func_kernel(const double *__restrict__ w, double *v, const double *__restrict__ u,
                              SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;
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
        v[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        v[linear_index] = 0.0;
        return;
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

__global__ void integrate_v_kernel(double *v_func, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[2] - 1)
        return;
    auto i = threadId / sim_params->dims[1];
    auto k = threadId % sim_params->dims[2];

    v_func[indexof(i, 0, k, sim_params)] = 0;
    for (unsigned int j = 1; j < sim_params->dims[1] - 1; j++) {
        if (i == 0) {
            v_func[indexof(i, j, k, sim_params)] = 0;
            continue;
        }
        if (i == sim_params->dims[0] - 1) {
            v_func[indexof(i, j, k, sim_params)] = 0;
            continue;
        }
        if (k == 0) {
            v_func[indexof(i, j, k, sim_params)] = 0;
            continue;
        }
        if (k == sim_params->dims[1] - 1) {
            v_func[indexof(i, j, k, sim_params)] = 0;
            continue;
        }
        v_func[indexof(i, j, k, sim_params)] =
                v_func[indexof(i, j - 1, k, sim_params)]
                - sim_params->deltas[1] * v_func[indexof(i, j, k, sim_params)];
    }
    v_func[indexof(i, sim_params->dims[1] - 1, k, sim_params)] = 0.0;
}

__global__ void w_kernel(const double *__restrict__ h, const double *__restrict__ old_w, const double *__restrict__ v,
                         double *w, SimulationParams *sim_params) {
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId > sim_params->dims[0] * sim_params->dims[1] * sim_params->dims[2] - 1)
        return;
    auto arr_ids      = indexof(threadId, sim_params);
    auto linear_index = indexof(arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    if (arr_ids.x == 0) {
        w[linear_index] = 0.0;
        return;
    }
    if (arr_ids.y == 0) {
        w[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == 0) {
        w[linear_index] = (sim_params->mins[1] + arr_ids.y * sim_params->deltas[1]) * c;
        return;
    }
    if (arr_ids.x == sim_params->dims[0] - 1) {
        w[linear_index] = 0.0;
        return;
    }
    if (arr_ids.y == sim_params->dims[1] - 1) {
        w[linear_index] = 0.0;
        return;
    }
    if (arr_ids.z == sim_params->dims[2] - 1) {
        w[linear_index] = 0.0;
        return;
    }

    Stencil3D w_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D h_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    Stencil3D v_point(old_w, arr_ids.x, arr_ids.y, arr_ids.z, sim_params);
    w[linear_index] = W_point(&h_point, &w_point, &v_point, sim_params);
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
