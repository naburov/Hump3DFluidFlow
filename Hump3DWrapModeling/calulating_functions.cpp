#define _USE_Mparams.ATH_DEFINES

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include "cell_calculating_functions."
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

double max_norm(double*** arr1, double*** arr0, SimulationParams params) {
	double max_norm = -1;
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t j = 0; j < params.dims[1]; j++)
			for (size_t k = 0; k < params.dims[2]; k++)
				if (std::abs(arr1[i][j][k] - arr0[i][j][k]) > max_norm)
					max_norm = std::abs(arr1[i][j][k] - arr0[i][j][k]);
	return max_norm;
}

double*** H(double*** U, SimulationParams params) {
	double*** next = new double** [params.dims[0]];
	for (size_t i = 0; i < params.dims[0]; i++) {
		next[i] = new double* [params.dims[1]];
		for (size_t j = 0; j < params.dims[1]; j++)
			next[i][j] = new double[params.dims[2]];
	}

	for (int i = 0; i < params.dims[0]; i++)
		for (int j = 0; j < params.dims[1]; j++)
			for (int k = 0; k < params.dims[2]; k++) {
				double xi1 = i * params.deltas[0] + params.mins[0];
				double xi2 = k * params.deltas[2] + params.mins[2];
				double theta = params.mins[1] + j * params.deltas[1];
				next[i][j][k] = U[i][j][k] - c * (theta + mu(xi1, xi2, params));
			}

	// boundary conditions
	for (size_t i = 0; i < params.dims[0]; i++) {
		double xi1 = params.mins[0] + i * params.deltas[0];
		for (size_t k = 0; k < params.dims[2]; k++) {
			double xi2 = params.mins[2] + k * params.deltas[2];
			next[i][0][k] = -c * mu(xi1, xi2, params);								// H| \theta = 0 == 0
			next[i][params.dims[1] - 1][k] = next[i][params.dims[1] - 2][k];				// H| \theta = \infty == 0
		}
	}

	double xi2 = params.mins[2];
	for (size_t i = 0; i < params.dims[0]; i++) {
		for (size_t j = 0; j < params.dims[1]; j++) {
			double xi1 = params.mins[0] + i * params.deltas[0];
			next[i][j][params.dims[2] - 1] = c * mu(xi1, xi2, params);			// H| \xi1,2->\infty = 0
			next[i][j][0] = c * mu(xi1, xi2, params);					// H| \xi1,2->-\infty = 0
		}
	}

	double xi1 = params.mins[0];
	for (size_t k = 0; k < params.dims[2]; k++) {
		for (size_t j = 0; j < params.dims[1]; j++) {
			double xi2 = params.mins[2] + k * params.deltas[2];
			next[params.dims[0] - 1][j][k] = c * mu(xi1, xi2, params);			// H| \xi1,2->\infty = 0
			next[0][j][k] = c * mu(xi1, xi2, params);					// H| \xi1,2->-\infty = 0
		}
	}
	return next;
}

double*** H(double*** prevH, double*** prevW, double*** prevV, SimulationParams params) {
	double*** next = new double** [params.dims[0]];
	for (size_t i = 0; i < params.dims[0]; i++) {
		next[i] = new double* [params.dims[1]];
		for (size_t j = 0; j < params.dims[1]; j++)
			next[i][j] = new double[params.dims[2]];
	}

#pragma omp parallel for
	for (int i = 0; i < params.dims[0] - 1; i++)
		for (int j = 0; j < params.dims[1] - 1; j++)
			for (int k = 0; k < params.dims[2] - 1; k++) {
				double xi1 = i * params.deltas[0] + params.mins[0];
				double xi2 = k * params.deltas[2] + params.mins[2];
				double theta = params.mins[1] + j * params.deltas[1];
				next[i][j][k] = prevH[i][j][k]
					- params.timeStep * (
						relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2, params))),
							derivative(prevH, 0, i, j, k, params.deltas[0], params.dims[0], params),
							derivative(prevH, 0, i + 1, j, k, params.deltas[0],params.dims[0], params))
						+ relaxed_derivative(
							(prevH[i][j][k] + c * (theta + mu(xi1, xi2, params))) * mu_derivative(xi1, xi2, 0, params) +
							prevW[i][j][k] * mu_derivative(xi1, xi2, 1, params) -
							prevV[i][j][k],
							derivative(prevH, 1, i, j, k, params.deltas[1],params.dims[2], params),
							derivative(prevH, 1, i, j + 1, k, params.deltas[1],params.dims[2], params))
						+ relaxed_derivative(
							prevW[i][j][k],
							derivative(prevH, 2, i, j, k, params.deltas[2],params.dims[1], params),
							derivative(prevH, 2, i, j, k + 1, params.deltas[2],params.dims[1], params))
						+ prevV[i][params.dims[1] - 1][k] * c
						+ prevV[i][j][k] * c
						- second_derivative(prevH, 1, i, j, k, params.deltas[1],params.dims[2], params)); // add p
			}

	// boundary conditions
	// boundary conditions
	for (size_t i = 0; i < params.dims[0]; i++) {
		double xi1 = params.mins[0] + i * params.deltas[0];
		for (size_t k = 0; k < params.dims[2]; k++) {
			double xi2 = params.mins[2] + k * params.deltas[2];
			next[i][0][k] = -c * mu(xi1, xi2, params);								// H| \theta = 0 == 0
			next[i][params.dims[1] - 1][k] = next[i][params.dims[1] - 2][k];				// H| \theta = \infty == 0
		}
	}

	double xi2 = params.mins[2];
	for (size_t i = 0; i < params.dims[0]; i++) {
		for (size_t j = 0; j < params.dims[1]; j++) {
			double xi1 = params.mins[0] + i * params.deltas[0];
			next[i][j][params.dims[2] - 1] = c * mu(xi1, xi2, params);			// H| \xi1,2->\infty = 0
			next[i][j][0] = c * mu(xi1, xi2, params);					// H| \xi1,2->-\infty = 0
		}
	}

	double xi1 = params.mins[0];
	for (size_t k = 0; k < params.dims[2]; k++) {
		for (size_t j = 0; j < params.dims[1]; j++) {
			double xi2 = params.mins[2] + k * params.deltas[2];
			next[params.dims[0] - 1][j][k] = c * mu(xi1, xi2, params);			// H| \xi1,2->\infty = 0
			next[0][j][k] = c * mu(xi1, xi2, params);					// H| \xi1,2->-\infty = 0
		}
	}
	return next;
}

double*** W(double*** prevH, double*** prevW, double*** prevV, SimulationParams params) {
	double*** next = new double** [params.dims[0]];
	for (size_t i = 0; i < params.dims[0]; i++) {
		next[i] = new double* [params.dims[1]];
		for (size_t j = 0; j < params.dims[1]; j++)
			next[i][j] = new double[params.dims[2]];

	}

#pragma omp parallel for
	for (int i = 0; i < params.dims[0] - 1; i++)
		for (int j = 0; j < params.dims[1] - 1; j++)
			for (int k = 0; k < params.dims[2] - 1; k++) {
				double xi1 = i * params.deltas[0] + params.mins[0];
				double xi2 = k * params.deltas[2] + params.mins[2];
				double theta = params.mins[1] + j * params.deltas[1];
				next[i][j][k] = prevW[i][j][k] - params.timeStep * (
					relaxed_derivative(
						prevW[i][j][k],
						derivative(prevW, 2, i, j, k, params.deltas[2],params.dims[1], params),
						derivative(prevW, 2, i, j, k + 1, params.deltas[2],params.dims[1], params)) +
					relaxed_derivative(
						prevH[i][j][k] + c * (theta + mu(xi1, xi2, params)),
						derivative(prevW, 0, i, j, k, params.deltas[0],params.dims[0], params),
						derivative(prevW, 0, i + 1, j, k, params.deltas[0],params.dims[0], params)) +
					relaxed_derivative(
						prevV[i][j][k] + mu_derivative(xi1, xi2, 1, params) * prevW[i][j][k]
						+ mu_derivative(xi1, xi2, 0, params) * (prevH[i][j][k] + c * (theta + mu(xi1, xi2, params))),
						derivative(prevW, 1, i, j, k, params.deltas[1],params.dims[2], params),
						derivative(prevW, 1, i, j + 1, k, params.deltas[1],params.dims[2], params)) -
					second_derivative(prevW, 1, i, j, k, params.deltas[1],params.dims[2], params)
					);
			}

	// boundary conditions
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t k = 0; k < params.dims[2]; k++) {
			next[i][0][k] = 0;					// W| \theta = 0 == 0
			next[i][params.dims[1] - 1][k] = 0;		// W| \theta = \infty == 0
		}

	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[i][j][params.dims[2] - 1] = 0;		// W| \xi1,2->\infty = 0
			next[i][j][0] = 0;					// W| \xi1,2->-\infty = 0
		}

	for (size_t k = 0; k < params.dims[2]; k++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[params.dims[0] - 1][j][k] = 0;		// W| \xi1,2->\infty = 0
			next[0][j][k] = 0;					// W| \xi1,2->-\infty = 0
		}

	return next;
}

double*** U(double*** H, SimulationParams params) {
	double*** next = new double** [params.dims[0]];
	for (size_t i = 0; i < params.dims[0]; i++) {
		next[i] = new double* [params.dims[1]];
		for (size_t j = 0; j < params.dims[1]; j++)
			next[i][j] = new double[params.dims[2]];
	}

#pragma omp parallel for
	for (int i = 0; i < params.dims[0]; i++)
		for (int j = 0; j < params.dims[1]; j++)
			for (int k = 0; k < params.dims[2]; k++) {
				double xi1 = i * params.deltas[0] + params.mins[0];
				double xi2 = k * params.deltas[2] + params.mins[2];
				double theta = params.mins[1] + j * params.deltas[1];
				next[i][j][k] = H[i][j][k] + c * (theta + mu(xi1, xi2, params));
			}

	// boundary conditions
	// U| \theta = 0 == 0
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t k = 0; k < params.dims[2]; k++)
			next[i][0][k] = 0;

	// U| \xi1,2->\infty = \theta * c
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[i][j][0] = (params.mins[1] + j * params.deltas[1]) * c;
			next[i][j][params.dims[2] - 1] = (params.mins[1] + j * params.deltas[1]) * c;
		}

	for (size_t k = 0; k < params.dims[2]; k++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[0][j][k] = (params.mins[1] + j * params.deltas[1]) * c;
			next[params.dims[0] - 1][j][k] = (params.mins[1] + j * params.deltas[1]) * c;
		}

	// dU| d\theta-> \infty == c
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t k = 0; k < params.dims[2]; k++) {
			next[i][params.dims[1] - 1][k] = c * params.deltas[1] + next[i][params.dims[1] - 2][k];
		}
	return next;
}

double integrate(const size_t& j, double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W, SimulationParams params)
{
	double sum = 0.0;
	for (int m = 1; m < j; m++) {
		int id1 = m;
		int id2 = m - 1;
		double theta1 = params.mins[1] + id2 * params.deltas[1];
		double theta2 = params.mins[1] + id1 * params.deltas[1];
		double f1 =
			central_derivative(U, 0, i, id1, k, params.deltas[0],params.dims[0], params) -
			mu_derivative(xi1, xi2, 0, params) *
			central_derivative(U, 1, i, id1, k, params.deltas[1],params.dims[2], params) +
			central_derivative(W, 2, i, id1, k, params.deltas[2],params.dims[1], params) -
			mu_derivative(xi1, xi2, 1, params) *
			central_derivative(W, 1, i, id1, k, params.deltas[1],params.dims[2], params);
		double f2 =
			central_derivative(U, 0, i, id2, k, params.deltas[0],params.dims[0], params) -
			mu_derivative(xi1, xi2, 0, params) *
			central_derivative(U, 1, i, id2, k, params.deltas[1],params.dims[2], params) +
			central_derivative(W, 2, i, id2, k, params.deltas[2],params.dims[1], params) -
			mu_derivative(xi1, xi2, 1, params) *
			central_derivative(W, 1, i, id2, k, params.deltas[1],params.dims[2], params);
		//if ((i == 1) && (j == 28))
		//	std::cout << "i=" << i << " m=" << m <<" " << "du/dxi1= " << derivative(U, 0, i, id1, k, params.deltas[0],params.dims[0]) << " " << derivative(U, 0, i + 1, id1, k, params.deltas[0],params.dims[0]) << std::endl;
		//if ((i == 50) && (j == 28))
		//	std::cout << "i=" << i << " m=" << m << " " << "du/dxi1= " << derivative(U, 0, i, id1, k, params.deltas[0],params.dims[0]) << " " << derivative(U, 0, i + 1, id1, k, params.deltas[0],params.dims[0]) << std::endl;
		sum -= 0.5 * params.deltas[1] * (f1 + f2);
	}
	return sum;
}

double v_func(int id1, double*** U, const size_t& i, const size_t& k, double xi1, double xi2, double*** W, SimulationParams params) {
	return	central_derivative(U, 0, i, id1, k, params.deltas[0],params.dims[0], params) -
		mu_derivative(xi1, xi2, 0, params) *
		central_derivative(U, 1, i, id1, k, params.deltas[1],params.dims[2], params) +
		central_derivative(W, 2, i, id1, k, params.deltas[2],params.dims[1], params) -
		mu_derivative(xi1, xi2, 1, params) *
		central_derivative(W, 1, i, id1, k, params.deltas[1],params.dims[2], params);
}

double*** V(double*** W, double*** U, SimulationParams params) {
	double*** next = new double** [params.dims[0]];
	for (size_t i = 0; i < params.dims[0]; i++) {
		next[i] = new double* [params.dims[1]];
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[i][j] = new double[params.dims[2]];
			std::fill_n(next[i][j], params.dims[2], 0.0);
		}
	}

#pragma omp parallel for
	for (int i = 1; i < params.dims[0] - 1; i++) {
		for (int k = 1; k < params.dims[2] - 1; k++) {

			for (int j = 1; j < params.dims[1] - 2; j++) {
				double xi1 = i * params.deltas[0] + params.mins[0];
				double xi2 = k * params.deltas[2] + params.mins[2];
				next[i][j][k] = next[i][j - 1][k] - params.deltas[1] * v_func(j, U, i, k, xi1, xi2, W, params);
			}
		}
	}

	// boundary condition
	// V| \theta = 0 == 0
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t k = 0; k < params.dims[2]; k++) {
			next[i][0][k] = 0.0;
			next[i][params.dims[1] - 1][k] = 0.0;
		}

	// V| \xi1,2->\infty = 0
	for (size_t i = 0; i < params.dims[0]; i++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[i][j][params.dims[2] - 1] = 0.0;
			next[i][j][0] = 0.0;
		}

	for (size_t k = 0; k < params.dims[2]; k++)
		for (size_t j = 0; j < params.dims[1]; j++) {
			next[params.dims[0] - 1][j][k] = 0.0;
			next[0][j][k] = 0.0;
		}

	return next;
}

double mu(double xi1, double xi2, SimulationParams params) {
	return params.A * std::exp(
		-std::pow(xi1, 2.0) * params.alpha -
		std::pow(xi2, 2.0) * params.beta);
}

double mu_derivative(double xi1, double xi2, int dim, SimulationParams params) {
	if (dim == 0)
		return -2 * params.A * xi1 * params.alpha * std::exp(-std::pow(xi1, 2.0) * params.alpha - std::pow(xi2, 2.0) * params.beta);
	else
		return -2 * params.A * xi2 * params.beta * std::exp(-std::pow(xi1, 2.0) * params.alpha - std::pow(xi2, 2.0) * params.beta);
}

double second_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params) {
	int lids[] = { i, j, k };
	lids[dim] -= 1;
	lids[0] = std::max(0, std::min(lids[0],params.dims[0] - 1));
	lids[1] = std::max(0, std::min(lids[1],params.dims[2] - 1));
	lids[2] = std::max(0, std::min(lids[2],params.dims[1] - 1));
	auto derivative = -2 * arr[i][j][k];
	derivative += arr[lids[0]][lids[1]][lids[2]];

	int rids[] = { i, j, k };
	rids[dim] += 1;
	rids[0] = std::max(0, std::min(rids[0],params.dims[0] - 1));
	rids[1] = std::max(0, std::min(rids[1],params.dims[2] - 1));
	rids[2] = std::max(0, std::min(rids[2],params.dims[1] - 1));

	return (derivative + arr[lids[0]][lids[1]][lids[2]]) / (delta * delta);
}

double central_derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params) {
	int ids[] = { i, j, k };
	ids[dim] -= 1;
	ids[0] = std::max(0, std::min(ids[0],params.dims[0] - 1));
	ids[1] = std::max(0, std::min(ids[1],params.dims[2] - 1));
	ids[2] = std::max(0, std::min(ids[2],params.dims[1] - 1));
	double derivative = -arr[ids[0]][ids[1]][ids[2]];

	int lids[] = { i, j, k };
	lids[dim] += 1;
	lids[0] = std::max(0, std::min(lids[0],params.dims[0] - 1));
	lids[1] = std::max(0, std::min(lids[1],params.dims[2] - 1));
	lids[2] = std::max(0, std::min(lids[2],params.dims[1] - 1));


	return (derivative + arr[lids[0]][lids[1]][lids[2]]) / (delta * 2);
}

double derivative(double*** arr, int dim, int i, int j, int k, double delta, int size, SimulationParams params) {
	int lids[] = { i, j, k };
	lids[dim] -= 1;

	lids[0] = std::max(0, std::min(lids[0], params.dims[0] - 1));
	lids[1] = std::max(0, std::min(lids[1], params.dims[2] - 1));
	lids[2] = std::max(0, std::min(lids[2], params.dims[1] - 1));

	i = std::max(0, std::min(i,params.dims[0] - 1));
	j = std::max(0, std::min(j,params.dims[2] - 1));
	k = std::max(0, std::min(k,params.dims[1] - 1));

	return (arr[i][j][k] - arr[lids[0]][lids[1]][lids[2]]) / delta;
}

double inline relaxed_derivative(double a, double derivative_left, double derivative_right) {
	return (a + std::abs(a)) / 2 * derivative_left + (a - std::abs(a)) / 2 * derivative_right;
}