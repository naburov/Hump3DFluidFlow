#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include "calculating_functions.h"
#include "consts.h"
#include <algorithm>
// 0 - x dimension in x
// 1 - y dimension in y
// 2 - z dimension in z

double max_norm(double*** arr1, double*** arr0, const int(&dims)[3]) {
	double max_norm = -1;
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t j = 0; j < dims[1]; j++)
			for (size_t k = 0; k < dims[2]; k++)
				if (std::abs(arr1[i][j][k] - arr0[i][j][k]) > max_norm)
					max_norm = std::abs(arr1[i][j][k] - arr0[i][j][k]);
	return max_norm;
}

double*** H(double*** U, const int(&dims)[3], const double(&deltas)[3]) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			next[i][j] = new double[dims[2]];

	}

	for (size_t i = 0; i < dims[0]; i++)
		for (size_t j = 0; j < dims[1]; j++)
			for (size_t k = 0; k < dims[2]; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = U[i][j][k] - c * (theta + mu(xi1, xi2));
			}


	return next;
}

double*** H(double*** prevH, double*** prevW, double*** prevV, const int(&dims)[3], const double(&deltas)[3], double timeStep) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			next[i][j] = new double[dims[2]];

	}

	for (size_t i = 1; i < dims[0] - 1; i++)
		for (size_t j = 1; j < dims[1] - 1; j++)
			for (size_t k = 1; k < dims[2] - 1; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = prevH[i][j][k]
					+ timeStep * (
						relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2))),
							derivative(prevH, 0, i, j, k, deltas[0]),
							derivative(prevH, 0, i + 1, j, k, deltas[0]))
						+ relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2))) * mu_derivative(xi1, xi2, 0) +
							prevW[i][j][k] * mu_derivative(xi1, xi2, 1) -
							prevV[i][j][k],
							derivative(prevH, 1, i, j, k, deltas[1]),
							derivative(prevH, 1, i, j + 1, k, deltas[1]))
						+ relaxed_derivative(
							prevW[i][j][k],
							derivative(prevH, 2, i, j, k, deltas[2]),
							derivative(prevH, 2, i, j, k + 1, deltas[2]))
						- prevV[i][dims[1] - 1][k] * c
						+ prevV[i][j][k] * c - second_derivative(prevH, 1, i, j, k, deltas[1])); // add p


			}
	return next;
}

double*** W(double*** prevH, double*** prevW, double*** prevV, const int(&dims)[3], const double(&deltas)[3], double timeStep) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			next[i][j] = new double[dims[2]];

	}

	std::vector<double> s;
	for (size_t i = 1; i < dims[0] - 1; i++)
		for (size_t j = 1; j < dims[1] - 1; j++)
			for (size_t k = 1; k < dims[2] - 1; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = prevW[i][j][k] + timeStep * (
					relaxed_derivative(
						prevW[i][j][k],
						derivative(prevW, 2, i, j, k, deltas[2]),
						derivative(prevW, 2, i, j, k + 1, deltas[2])) +
					relaxed_derivative(
						prevH[i][j][k] + c * (theta + mu(xi1, xi2)),
						derivative(prevW, 0, i, j, k, deltas[0]),
						derivative(prevW, 0, i + 1, j, k, deltas[0])) +
					relaxed_derivative(
						prevV[i][j][k] + mu_derivative(xi1, xi2, 1) * prevW[i][j][k]
						+ mu_derivative(xi1, xi2, 0) * (prevH[i][j][k] + c * (theta + mu(xi1, xi2))),
						derivative(prevW, 1, i, j, k, deltas[1]),
						derivative(prevW, 1, i, j + 1, k, deltas[1])) -
					second_derivative(prevW, 1, i, j, k, deltas[1])
					);
				s.push_back(next[i][j][k]);
			}
	std::cout << "w max " << *std::max_element(s.begin(), s.end()) << std::endl;

	// boundary conditions
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++) {
			next[i][0][k] = 0;					// W| \theta = 0 == 0
			next[i][dims[1] - 1][k] = 0;		// W| \theta = \infty == 0
		}

	for (size_t i = 0; i < dims[0]; i++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[i][j][dims[2] - 1] = 0;		// W| \xi1,2->\infty = 0
			next[i][j][0] = 0;					// W| \xi1,2->-\infty = 0
		}

	for (size_t k = 0; k < dims[2]; k++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[dims[0] - 1][j][k] = 0;		// W| \xi1,2->\infty = 0
			next[0][j][k] = 0;					// W| \xi1,2->-\infty = 0
		}

	return next;
}

double*** U(double*** H, const int(&dims)[3], const double(&deltas)[3]) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			next[i][j] = new double[dims[2]];

	}
	std::vector<double> s;
	for (size_t i = 1; i < dims[0] - 1; i++)
		for (size_t j = 1; j < dims[1] - 1; j++)
			for (size_t k = 1; k < dims[2] - 1; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = H[i][j][k] + c * (theta + mu(xi1, xi2));
				s.push_back(next[i][j][k]);
			}
	std::cout << "u max " << *std::max_element(s.begin(), s.end()) << std::endl;

	// boundary conditions
	// U| \theta = 0 == 0
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++)
			next[i][0][k] = 0;

	// dU| d\theta-> \infty == c
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++) {
			next[i][dims[1] - 1][k] = c * deltas[1] + next[i][dims[1] - 2][k];
		}

	// U| \xi1,2->\infty = \theta * c
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[i][j][0] = (theta_min + j * deltas[1]) * c;
			next[i][j][dims[2] - 1] = (theta_min + j * deltas[1]) * c;
		}

	for (size_t k = 0; k < dims[2]; k++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[0][j][k] = (theta_min + j * deltas[1]) * c;
			next[dims[0] - 1][j][k] = (theta_min + j * deltas[1]) * c;
		}
	return next;
}

double*** V(double*** W, double*** U, const int(&dims)[3], const double(&deltas)[3]) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			next[i][j] = new double[dims[2]];

	}

	std::vector<double> s;

	for (size_t i = 1; i < dims[0] - 1; i++)
		for (size_t k = 1; k < dims[2] - 1; k++)
			for (size_t j = 1; j < dims[1] - 1; j++) {
				// integrating cycle
				long double sum = 0;
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				for (size_t m = 1; m < j - 1; m++) {
					double theta1 = theta_min + m * deltas[1];
					double theta2 = theta_min + (m + 1) * deltas[1];
					double f1 = derivative(U, 0, i, m, k, deltas[0]) -
						mu_derivative(xi1, xi2, 0) * derivative(U, 1, i, m, k, deltas[1]) +
						derivative(W, 2, i, m, k, deltas[2]) -
						mu_derivative(xi1, xi2, 1) * derivative(W, 1, i, m, k, deltas[1]);
					double f2 = derivative(U, 0, i, m + 1, k, deltas[0]) -
						mu_derivative(xi1, xi2, 0) * derivative(U, 1, i, m + 1, k, deltas[1]) +
						derivative(W, 2, i, m + 1, k, deltas[2]) -
						mu_derivative(xi1, xi2, 1) * derivative(W, 1, i, m + 1, k, deltas[1]);
					sum -= 0.5 * (theta2 - theta1) * (f1 + f2);
				}
				s.push_back(sum);
				//std::cout << sum << std::endl;
				next[i][j][k] = sum;
			}
	std::cout << "v max " << *std::max_element(s.begin(), s.end()) << std::endl;

	// boundary condition
	// V| \theta = 0 == 0
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 1; k < dims[2]; k++) {
			next[i][0][k] = 0.0;
			next[i][dims[1] - 1][k] = 0.0;
		}

	// V| \xi1,2->\infty = 0
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[i][j][dims[2] - 1] = 0.0;
			next[i][j][0] = 0.0;
		}

	for (size_t k = 0; k < dims[2]; k++)
		for (size_t j = 0; j < dims[1]; j++) {
			next[dims[0] - 1][j][k] = 0.0;
			next[0][j][k] = 0.0;
		}

	return next;
}

double mu(double xi1, double xi2) {
	return 1 * std::exp(
		-std::pow(xi1, 2.0) * alpha -
		std::pow(xi2, 2.0) * beta);
}

double mu_derivative(double xi1, double xi2, int dim) {
	if (dim == 0)
		return 1 * xi1 * std::exp(-std::pow(xi1, 2.0) * alpha - std::pow(xi2, 2.0) * beta) * alpha;
	else
		return 1 * xi2 * std::exp(-std::pow(xi1, 2.0) * alpha - std::pow(xi2, 2.0) * beta) * beta;
}

double second_derivative(double*** arr, int dim, int i, int j, int k, double delta) {
	int ids[] = { i, j, k };
	ids[dim] -= 1;
	double derivative = -2 * arr[i][j][k] + arr[ids[0]][ids[1]][ids[2]];
	ids[dim] += 2;
	return (derivative + arr[ids[0]][ids[1]][ids[2]]) / (delta * delta);
}

double derivative(double*** arr, int dim, int i, int j, int k, double delta) {
	int ids[] = { i, j, k };
	ids[dim] -= 1;
	return (arr[i][j][k] - arr[ids[0]][ids[1]][ids[2]]) / delta;
}

double relaxed_derivative(double a, double derivative_left, double derivative_right) {
	return (a + std::abs(a)) / 2 * derivative_left + (a - std::abs(a)) / 2 * derivative_right;
}