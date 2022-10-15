#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include "calculating_functions.h"
#include "consts.h"
#include <algorithm>
#include <omp.h>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <iomanip>
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

	for (int i = 0; i < dims[0]; i++)
		for (int j = 0; j < dims[1]; j++)
			for (int k = 0; k < dims[2]; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = U[i][j][k] - c * (theta + mu(xi1, xi2));
			}

	// boundary conditions
	for (size_t i = 0; i < dims[0]; i++) {
		double xi1 = xi1_min + i * deltas[0];
		for (size_t k = 0; k < dims[2]; k++) {
			double xi2 = xi2_min + k * deltas[2];
			next[i][0][k] = -c * mu(xi1, xi2);								// H| \theta = 0 == 0
			next[i][dims[1] - 1][k] = next[i][dims[1] - 2][k];				// H| \theta = \infty == 0
		}
	}

	double xi2 = xi2_min;
	for (size_t i = 0; i < dims[0]; i++) {
		for (size_t j = 0; j < dims[1]; j++) {
			double xi1 = xi1_min + i * deltas[0];
			next[i][j][dims[2] - 1] = c * mu(xi1, xi2);			// H| \xi1,2->\infty = 0
			next[i][j][0] = c * mu(xi1, xi2);					// H| \xi1,2->-\infty = 0
		}
	}

	double xi1 = xi1_min;
	for (size_t k = 0; k < dims[2]; k++) {
		for (size_t j = 0; j < dims[1]; j++) {
			double xi2 = xi2_min + k * deltas[2];
			next[dims[0] - 1][j][k] = c * mu(xi1, xi2);			// H| \xi1,2->\infty = 0
			next[0][j][k] = c * mu(xi1, xi2);					// H| \xi1,2->-\infty = 0
		}
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

#pragma omp parallel for
	for (int i = 0; i < dims[0] - 1; i++)
		for (int j = 0; j < dims[1] - 1; j++)
			for (int k = 0; k < dims[2] - 1; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = prevH[i][j][k]
					- timeStep * (
						relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2))),
							derivative(prevH, 0, i, j, k, deltas[0], N),
							derivative(prevH, 0, i + 1, j, k, deltas[0], N))
						+ relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2))) * mu_derivative(xi1, xi2, 0) +
							prevW[i][j][k] * mu_derivative(xi1, xi2, 1) -
							prevV[i][j][k],
							derivative(prevH, 1, i, j, k, deltas[1], M),
							derivative(prevH, 1, i, j + 1, k, deltas[1], M))
						+ relaxed_derivative(
							prevW[i][j][k],
							derivative(prevH, 2, i, j, k, deltas[2], K),
							derivative(prevH, 2, i, j, k + 1, deltas[2], K))
						+ prevV[i][dims[1] - 1][k] * c
						+ prevV[i][j][k] * c
						- second_derivative(prevH, 1, i, j, k, deltas[1], M)); // add p
			}

	// boundary conditions
	// boundary conditions
	for (size_t i = 0; i < dims[0]; i++) {
		double xi1 = xi1_min + i * deltas[0];
		for (size_t k = 0; k < dims[2]; k++) {
			double xi2 = xi2_min + k * deltas[2];
			next[i][0][k] = -c * mu(xi1, xi2);								// H| \theta = 0 == 0
			next[i][dims[1] - 1][k] = next[i][dims[1] - 2][k];				// H| \theta = \infty == 0
		}
	}

	double xi2 = xi2_min;
	for (size_t i = 0; i < dims[0]; i++) {
		for (size_t j = 0; j < dims[1]; j++) {
			double xi1 = xi1_min + i * deltas[0];
			next[i][j][dims[2] - 1] = c * mu(xi1, xi2);			// H| \xi1,2->\infty = 0
			next[i][j][0] = c * mu(xi1, xi2);					// H| \xi1,2->-\infty = 0
		}
	}

	double xi1 = xi1_min;
	for (size_t k = 0; k < dims[2]; k++) {
		for (size_t j = 0; j < dims[1]; j++) {
			double xi2 = xi2_min + k * deltas[2];
			next[dims[0] - 1][j][k] = c * mu(xi1, xi2);			// H| \xi1,2->\infty = 0
			next[0][j][k] = c * mu(xi1, xi2);					// H| \xi1,2->-\infty = 0
		}
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

#pragma omp parallel for
	for (int i = 0; i < dims[0] - 1; i++)
		for (int j = 0; j < dims[1] - 1; j++)
			for (int k = 0; k < dims[2] - 1; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = prevW[i][j][k] - timeStep * (
					relaxed_derivative(
						prevW[i][j][k],
						derivative(prevW, 2, i, j, k, deltas[2], K),
						derivative(prevW, 2, i, j, k + 1, deltas[2], K)) +
					relaxed_derivative(
						prevH[i][j][k] + c * (theta + mu(xi1, xi2)),
						derivative(prevW, 0, i, j, k, deltas[0], N),
						derivative(prevW, 0, i + 1, j, k, deltas[0], N)) +
					relaxed_derivative(
						prevV[i][j][k] + mu_derivative(xi1, xi2, 1) * prevW[i][j][k]
						+ mu_derivative(xi1, xi2, 0) * (prevH[i][j][k] + c * (theta + mu(xi1, xi2))),
						derivative(prevW, 1, i, j, k, deltas[1], M),
						derivative(prevW, 1, i, j + 1, k, deltas[1], M)) -
					second_derivative(prevW, 1, i, j, k, deltas[1], M)
					);
			}

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

#pragma omp parallel for
	for (int i = 0; i < dims[0]; i++)
		for (int j = 0; j < dims[1]; j++)
			for (int k = 0; k < dims[2]; k++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				double theta = theta_min + j * deltas[1];
				next[i][j][k] = H[i][j][k] + c * (theta + mu(xi1, xi2));
			}

	// boundary conditions
	// U| \theta = 0 == 0
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++)
			next[i][0][k] = 0;

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

	// dU| d\theta-> \infty == c
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++) {
			next[i][dims[1] - 1][k] = c * deltas[1] + next[i][dims[1] - 2][k];
		}
	return next;
}

double integrate(const size_t& j, const double(&deltas)[3], double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W)
{
	double sum = 0.0;
	for (int m = 1; m < j; m++) {
		int id1 = m;
		int id2 = m - 1;
		double theta1 = theta_min + id2 * deltas[1];
		double theta2 = theta_min + id1 * deltas[1];
		double f1 =
			central_derivative(U, 0, i, id1, k, deltas[0], N) -
			mu_derivative(xi1, xi2, 0) *
			central_derivative(U, 1, i, id1, k, deltas[1], M) +
			central_derivative(W, 2, i, id1, k, deltas[2], K) -
			mu_derivative(xi1, xi2, 1) *
			central_derivative(W, 1, i, id1, k, deltas[1], M);
		double f2 =
			central_derivative(U, 0, i, id2, k, deltas[0], N) -
			mu_derivative(xi1, xi2, 0) *
			central_derivative(U, 1, i, id2, k, deltas[1], M) +
			central_derivative(W, 2, i, id2, k, deltas[2], K) -
			mu_derivative(xi1, xi2, 1) *
			central_derivative(W, 1, i, id2, k, deltas[1], M);
		//if ((i == 1) && (j == 28))
		//	std::cout << "i=" << i << " m=" << m <<" " << "du/dxi1= " << derivative(U, 0, i, id1, k, deltas[0], N) << " " << derivative(U, 0, i + 1, id1, k, deltas[0], N) << std::endl;
		//if ((i == 50) && (j == 28))
		//	std::cout << "i=" << i << " m=" << m << " " << "du/dxi1= " << derivative(U, 0, i, id1, k, deltas[0], N) << " " << derivative(U, 0, i + 1, id1, k, deltas[0], N) << std::endl;
		sum -= 0.5 * deltas[1] * (f1 + f2);
	}
	return sum;
}

double v_func(int id1, const double(&deltas)[3], double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W) {
	return	central_derivative(U, 0, i, id1, k, deltas[0], N) -
		mu_derivative(xi1, xi2, 0) *
		central_derivative(U, 1, i, id1, k, deltas[1], M) +
		central_derivative(W, 2, i, id1, k, deltas[2], K) -
		mu_derivative(xi1, xi2, 1) *
		central_derivative(W, 1, i, id1, k, deltas[1], M);
}

double*** V(double*** W, double*** U, const int(&dims)[3], const double(&deltas)[3]) {
	double*** next = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		next[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++) {
			next[i][j] = new double[dims[2]];
			std::fill_n(next[i][j], dims[2], 0.0);
		}
	}

#pragma omp parallel for
	for (int i = 1; i < dims[0] - 1; i++) {
		for (int k = 1; k < dims[2] - 1; k++) {

			for (int j = 1; j < dims[1] - 2; j++) {
				double xi1 = i * deltas[0] + xi1_min;
				double xi2 = k * deltas[2] + xi2_min;
				next[i][j][k] = next[i][j - 1][k] - deltas[1] * v_func(j, deltas, U, i, k, xi1, xi2, W);
			}
		}
	}

	// boundary condition
	// V| \theta = 0 == 0
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++) {
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
	return A * std::exp(
		-std::pow(xi1, 2.0) * alpha -
		std::pow(xi2, 2.0) * beta);
}

double mu_derivative(double xi1, double xi2, int dim) {
	if (dim == 0)
		return -2 * A * xi1 * alpha * std::exp(-std::pow(xi1, 2.0) * alpha - std::pow(xi2, 2.0) * beta);
	else
		return -2 * A * xi2 * beta * std::exp(-std::pow(xi1, 2.0) * alpha - std::pow(xi2, 2.0) * beta);
}

double second_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size) {
	int lids[] = { i, j, k };
	lids[dim] -= 1;
	lids[0] = std::max(0, std::min(lids[0], N - 1));
	lids[1] = std::max(0, std::min(lids[1], M - 1));
	lids[2] = std::max(0, std::min(lids[2], K - 1));
	auto derivative = -2 * arr[i][j][k];
	derivative += arr[lids[0]][lids[1]][lids[2]];

	int rids[] = { i, j, k };
	rids[dim] += 1;
	rids[0] = std::max(0, std::min(rids[0], N - 1));
	rids[1] = std::max(0, std::min(rids[1], M - 1));
	rids[2] = std::max(0, std::min(rids[2], K - 1));

	return (derivative + arr[lids[0]][lids[1]][lids[2]]) / (delta * delta);
}

double central_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size) {
	int ids[] = { i, j, k };
	ids[dim] -= 1;
	ids[0] = std::max(0, std::min(ids[0], N - 1));
	ids[1] = std::max(0, std::min(ids[1], M - 1));
	ids[2] = std::max(0, std::min(ids[2], K - 1));
	double derivative = -arr[ids[0]][ids[1]][ids[2]];

	int lids[] = { i, j, k };
	lids[dim] += 1;
	lids[0] = std::max(0, std::min(lids[0], N - 1));
	lids[1] = std::max(0, std::min(lids[1], M - 1));
	lids[2] = std::max(0, std::min(lids[2], K - 1));


	return (derivative + arr[lids[0]][lids[1]][lids[2]]) / (delta * 2);
}

double derivative(double*** arr, int dim, int i, int j, int k, double delta, int size) {
	int lids[] = { i, j, k };
	lids[dim] -= 1;

	lids[0] = std::max(0, std::min(lids[0], N - 1));
	lids[1] = std::max(0, std::min(lids[1], M - 1));
	lids[2] = std::max(0, std::min(lids[2], K - 1));

	i = std::max(0, std::min(i, N - 1));
	j = std::max(0, std::min(j, M - 1));
	k = std::max(0, std::min(k, K - 1));

	return (arr[i][j][k] - arr[lids[0]][lids[1]][lids[2]]) / delta;
}

double inline relaxed_derivative(double a, double derivative_left, double derivative_right) {
	return (a + std::abs(a)) / 2 * derivative_left + (a - std::abs(a)) / 2 * derivative_right;
}