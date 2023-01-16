#include "SimulationParams.h"

__device__ int indexof(int x, int y, int z, const SimulationParams &params) {
    return x * params.dims[0] + y * params.dims[1] + z;
}

//__device__ double relaxed_derivative(double a, double derivative_left, double derivative_right) {
//    return derivative_left * (a + abs(a)) / 2 +
//           derivative_right * (a - abs(a)) / 2;
//}
//
//__device__ double mu(double xi1, double xi2, const SimulationParams &params) {
//    return params.A * exp(-pow(xi1, 2.0) * params.alpha
//                          - pow(xi2, 2.0) * params.beta);
//}
//
//__device__ double mu_derivative(double xi1, double xi2, int dim, const SimulationParams &params) {
//    if (dim == 0)
//        return -2 * params.A * xi1 * params.alpha * exp(-pow(xi1, 2.0) * params.alpha
//                                                        - pow(xi2, 2.0) * params.beta);
//    else
//        return -2 * params.A * xi2 * params.beta * exp(-pow(xi1, 2.0) * params.alpha
//                                                       - pow(xi2, 2.0) * params.beta);
//}
//
//__device__ double
//u_point(double h_point, double c, double theta, double xi1, double xi2, const SimulationParams &params) {
//    return h_point + c * (theta + mu(xi1, xi2, params));
//}
//
//__device__ double
//H_point(double u_point, double c, double theta, double xi1, double xi2, const SimulationParams &params) {
//    return u_point - c * (theta + mu(xi1, xi2, params));
//}
//
//__device__ double left_derivative(double left, double right, double delta) {
//    return (right - left) / delta;
//}
//
//__device__ double H_point(double h_point, double w_point, double v_point, double xi1, double x2, double theta,
//                          const SimulationParams &params) {
//    return h_point - params.timeStep;
//}