

#include "SimulationParams.h"
#include "3D_stencil.cuh"
#include "cuda_consts.cuh"


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

//tanh
__device__ __host__ double sech(double x) {
    return 1. / cosh(x);
}

__device__ __host__ double mu(double xi1, double xi2, SimulationParams *params) {
//    return 0.0;
    return params->A * 1. / 4 *
           (-tanh(1. / 2 * (xi1 - 2) * (xi1 + 2)) + 1) *
           (-tanh(1. / 2 * (xi2 - 2) * (xi2 + 2)) + 1);
}

__device__ __host__ double mu_derivative(double xi1, double xi2, int dim, SimulationParams *params) {
//    return 0.0;
    if (dim == 0) {
        auto t  = sech(1. / 2 * (xi1 - 2) * (xi1 + 2)) * sech(1. / 2 * (xi1 - 2) * (xi1 + 2));
        auto t1 = 1. / 2 * (xi1 - 2) + 1. / 2 * (xi1 + 2);
        return -1. / 4 * t1 * t * (1 - tanh(1. / 2 * (xi2 + 2) * (xi2 - 2))) * params->A;
    } else {
        auto t  = sech(1. / 2 * (xi2 - 2) * (xi2 + 2)) * sech(1. / 2 * (xi2 - 2) * (xi2 + 2));
        auto t1 = 1. / 2 * (xi2 - 2) + 1. / 2 * (xi2 + 2);
        return -1. / 4 * t1 * t * (1 - tanh(1. / 2 * (xi1 + 2) * (xi1 - 2))) * params->A;
    }
}

// exponential
//__device__ __host__ double mu(double xi1, double xi2, SimulationParams *params) {
//    return params->A * exp(-pow(xi1, 2.0) * params->alpha
//                           - pow(xi2, 2.0) * params->beta);
//}

//__device__ __host__ double mu_derivative(double xi1, double xi2, int dim, SimulationParams *params) {
//    if (dim == 0)
//        return -2. * params->A * xi1 * params->alpha * exp(-pow(xi1, 2.0) * params->alpha
//                                                           - pow(xi2, 2.0) * params->beta);
//    else
//        return -2. * params->A * xi2 * params->beta * exp(-pow(xi1, 2.0) * params->alpha
//                                                          - pow(xi2, 2.0) * params->beta);
//}

__device__ double
U_point(Stencil3D *__restrict__ H, SimulationParams *params) {
    return H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params));
}

__device__ double
H_point(Stencil3D *__restrict__ U, SimulationParams *params) {
    return U->center.w - c * (U->center.y + mu(U->center.x, U->center.z, params));
}

__device__ double
H_point(Stencil3D *__restrict__ H, Stencil3D *__restrict__ W, Stencil3D *__restrict__ V, const double dp,
        SimulationParams *params) {
    auto theta = H->center.y;
    auto xi1   = H->center.x;
    auto xi2   = H->center.z;
    // H->center.x = xi1
    // H->center.z = xi2
    return H->center.w
           - params->timeStep * (
            relaxed_derivative(
                    H->center.w + c * (theta + mu(xi1, xi2, params)),
                    H->dx_l(),
                    H->dx_r())
            - relaxed_derivative(
                    (H->center.w + c * (theta + mu(xi1, xi2, params))) *
                    mu_derivative(xi1, xi2, 0, params) +
                    W->center.w * mu_derivative(xi1, xi2, 1, params) -
                    V->center.w,
                    H->dy_l(),
                    H->dy_r())
            + relaxed_derivative(
                    W->center.w,
                    H->dz_l(),
                    H->dz_r())
            + dp * c
            + V->center.w * c
            - H->dy2()
    );
}

// {w_{i, j, k}^{\dagger}}^{t+1} =& w_{i, j, k}^{\dagger} + \Delta t \bigg(
// {[w_{i, j, k}^{\dagger}]}^+\bigg(\dfrac{w_{i, j, k}^{\dagger} - w_{i, j, k-1}^{\dagger}}{h_z} \bigg)
// + {[w_{i, j, k}^{\dagger}]}^-\bigg(\dfrac{w_{i, j, k+1}^{\dagger}
// - w_{i, j, k}^{\dagger}}{h_z} \bigg) + \\ & +
// \big[H_{i, j, k} + c(\theta + \mu)\big]^+ \bigg(\dfrac{w_{i, j, k}^{\dagger} - w_{i-1, j, k}^{\dagger}}{h_x}\bigg) +
// +  \big[H_{i, j, k} + c(\theta + \mu)]^- \bigg(\dfrac{w_{i+1, j, k}^{\dagger} - w_{i, j, k}^{\dagger}}{h_x}\bigg) +
// + {\big[ v_{i, j, k}^{\dagger} + w_{i, j, k}^{\dagger}\frac{\partial \mu_{i, j, k}}{\partial \xi_2}
// + \frac{\partial \mu_{i, j, k}}{\partial \xi_2}(H_{i, j, k}
// + c(\theta + \mu_{i, j, k}))\big]}^+\bigg(\dfrac{w_{i, j, k}^{\dagger} - w_{i, j-1, k}^{\dagger}}{h_y} \bigg) +
// + {\big[ v_{i, j, k}^{\dagger} + w_{i, j, k}^{\dagger}\frac{\partial \mu_{i, j, k}}{\partial \xi_2}
// + \frac{\partial \mu_{i, j, k}}{\partial \xi_2}(H_{i, j, k} +
// + c(\theta + \mu_{i, j, k}))\big]}^-\bigg(\dfrac{w_{i, j+1, k}^{\dagger} - w_{i, j, k}^{\dagger}}{h_y} \bigg)
// - \bigg[\dfrac{w_{i, j, k+1}^{\dagger} - w_{i, j, k}^{\dagger}}{{h_z}^2} \big]
//\bigg),
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
            - relaxed_derivative(
                    W->center.w * mu_derivative(H->center.x, H->center.z, 1, params) +
                    mu_derivative(H->center.x, H->center.z, 0, params) *
                    (H->center.w + c * (H->center.y + mu(H->center.x, H->center.z, params)))
                    - V->center.w,
                    W->dy_l(),
                    W->dy_r())
            - W->dy2());
}
// v^{\dagger} =- \int_{0}^{\theta}\bigg(\dfrac{\partial u^{\dagger}}{\partial \xi_1} -
// \dfrac{\partial \mu}{\partial \xi_1}\dfrac{\partial u^{\dagger}}{\partial \theta} +
// \dfrac{\partial w^{\dagger}}{\partial \xi_2} -
// \dfrac{\partial \mu}{\partial \xi_2}\dfrac{\partial w^{\dagger}}{\partial \theta} \bigg)d\theta'
__device__ double v_func(Stencil3D *__restrict__ U, Stencil3D *__restrict__ W, SimulationParams *params) {
    return U->dx_r()
           - mu_derivative(U->center.x, U->center.z, 0, params) * U->dy_r()
           + W->dz_r()
           - mu_derivative(U->center.x, U->center.z, 1, params) * W->dy_r();
}

