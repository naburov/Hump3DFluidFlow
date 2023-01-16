#pragma once

#include "SimulationParams.h"

double max_norm(double*** arr1, double*** arr0, SimulationParams params);
double*** H(double*** U, SimulationParams params);
double*** W(double*** prevH, double*** prevW, double*** prevV, SimulationParams params);
double*** U(double*** H, SimulationParams params);
double integrate(const size_t& j, double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W, SimulationParams params);
double*** V(double*** W, double*** U, SimulationParams params);
double mu(double xi1, double xi2, SimulationParams params);
double mu_derivative(double xi1, double xi2, int dim, SimulationParams params);
double second_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params);
double derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params);
double relaxed_derivative(double a, double derivative_left, double derivative_right);
double*** H(double*** prevH, double*** prevW, double*** prevV, SimulationParams params);
double central_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params);