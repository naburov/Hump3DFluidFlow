#pragma once
double max_norm(double*** arr1, double*** arr0, const int(&dims)[3]);
double*** H(double*** prevH, double*** prevW, double*** prevV, const int(&dims)[3], const double(&deltas)[3], double timeStep);
double*** W(double*** prevH, double*** prevW, double*** prevV, const int(&dims)[3], const double(&deltas)[3], double timeStep);
double*** U(double*** H, const int(&dims)[3], const double(&deltas)[3]);
double integrate(const size_t& j, const double(&deltas)[3], double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W);
double*** V(double*** W, double*** U, const int(&dims)[3], const double(&deltas)[3]);
double mu(double xi1, double xi2);
double mu_derivative(double xi1, double xi2, int dim);
double second_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size);
double derivative(double*** arr, int dim, int i, int j, int k, double delta, int size);
double relaxed_derivative(double a, double derivative_left, double derivative_right);
double*** H(double*** U, const int(&dims)[3], const double(&deltas)[3]);
double central_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size);
