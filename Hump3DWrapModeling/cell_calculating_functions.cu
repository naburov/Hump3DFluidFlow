#include "SimulationParams.h"
#include "3D_stencil.cuh"
#include "consts.h"


__device__ __host__ unsigned int indexof(unsigned int x, unsigned int y, unsigned int z, SimulationParams *params) {
    return x * (params->dims[0] * params->dims[1]) + y * params->dims[1] + z;
}

__device__ __host__ int3 indexof(unsigned int id, SimulationParams *params) {
    int3 res;
    res.x = id / (params->dims[0] * params->dims[1]);
    res.z = id % params->dims[1];
    res.y = (id - res.x * (params->dims[0] * params->dims[1]) - res.z) / params->dims[1];
    return res;
}

__device__ double relaxed_derivative(double a, double derivative_left, double derivative_right) {
    return derivative_left * (a + abs(a)) / 2 +
           derivative_right * (a - abs(a)) / 2;
}

__device__ double mu(double xi1, double xi2, SimulationParams *params) {
    return params->A * exp(-pow(xi1, 2.0) * params->alpha
                           - pow(xi2, 2.0) * params->beta);
}

__device__ double mu_derivative(double xi1, double xi2, int dim, SimulationParams *params) {
    if (dim == 0)
        return -2 * params->A * xi1 * params->alpha * exp(-pow(xi1, 2.0) * params->alpha
                                                          - pow(xi2, 2.0) * params->beta);
    else
        return -2 * params->A * xi2 * params->beta * exp(-pow(xi1, 2.0) * params->alpha
                                                         - pow(xi2, 2.0) * params->beta);
}

__device__ double
U_point(Stencil3D *__restrict__ H, SimulationParams *params) {
    return H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params));
}

__device__ double
H_point(Stencil3D *__restrict__ U, SimulationParams *params) {
    return U->center.w - c * (U->center.y + mu(U->center.x, U->center.z, params));
}

__device__ double
H_point(Stencil3D *__restrict__ H, Stencil3D *__restrict__ W, Stencil3D *__restrict__ V, SimulationParams *params) {
    // H->center.y = theta
    // H->center.x = xi1
    // H->center.z = xi2
    return H->center.w
           - params->timeStep * (
            relaxed_derivative(
                    H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params)),
                    H->dx_l(),
                    H->dx_r())
            + relaxed_derivative(
                    (H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params))) *
                    mu_derivative(H->center.x, H->center.z, 0, params) +
                    W->center.w * mu_derivative(H->center.x, H->center.z, 1, params) - V->center.w,
                    H->dy_l(),
                    H->dy_r())
            + relaxed_derivative(
                    W->center.w,
                    H->dz_l(),
                    H->dz_r())
            + V->y_less.w * c + V->center.w * c -
            H->dy2()
    );
}

__device__ double
W_point(Stencil3D *__restrict__ H, Stencil3D *__restrict__ W, Stencil3D *__restrict__ V, SimulationParams *params) {
    return W->center.w
           - params->timeStep * (
            relaxed_derivative(
                    W->center.w,
                    W->dz_l(),
                    W->dz_r())
            + relaxed_derivative(
                    H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params)),
                    W->dx_l(),
                    W->dx_r())
            + relaxed_derivative(
                    V->center.w + W->center.w * mu_derivative(H->center.x, H->center.z, 1, params) +
                    mu_derivative(H->center.x, H->center.z, 0, params) *
                    (H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params))),
                    W->dy_l(),
                    W->dy_r())
            - W->dy2());
}

__device__ double v_func(Stencil3D *__restrict__ U, Stencil3D *__restrict__ W, SimulationParams *params) {
    return U->dx_c()
           - mu_derivative(U->center.x, U->center.z, 0, params) * U->dy_c()
           + W->dz_c()
           - mu_derivative(U->center.x, U->center.z, 1, params) * W->dy_c();
}

