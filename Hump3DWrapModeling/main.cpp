#define _CRT_SECURE_NO_WARNINGS

#include <iomanip>
#include <iostream>
#include <sstream>  
#include <omp.h>

#include "consts.h"
#include "export_functions.h"
#include "calculating_functions.h"
#include <vector>
#include <algorithm>
#include <direct.h>
#include <filesystem>

int main() {
	omp_set_num_threads(8);

	const int dims[] = { N, M, K };
	const double deltas[] = { (xi1_max - xi1_min) / (N - 1), (theta_max - theta_min) / (M - 1), (xi2_max - xi2_min) / (K - 1) };
	const double time_step = static_cast<double>(tmax) / T;

	// initalize U
	double*** u = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		u[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++)
			u[i][j] = new double[dims[2]];

	}

	std::vector<double> s;
	for (size_t i = 0; i < dims[0]; i++)
		for (size_t k = 0; k < dims[2]; k++)
		{
			double xi1 = i * deltas[0] + xi1_min;
			double xi2 = k * deltas[2] + xi2_min;
			int j = 0;
			for (j = 0; theta_min + deltas[1] * (double)j < 5.0; j++) {
				double theta = theta_min + j * deltas[1];
				u[i][j][k] = f_second * theta * (1 + 0.2 * mu(xi1, xi2));
				s.push_back(u[i][j][k]);
			}

			for (j; j < dims[1]; j++) {
				double theta = theta_min + j * deltas[1];
				u[i][j][k] = f_second * (theta + mu(xi1, xi2));
				s.push_back(u[i][j][k]);
			}

		}
	//std::cout << "u max " << *std::max_element(s.begin(), s.end()) << std::endl;

	// initialize W
	double*** w = new double** [dims[0]];
	for (size_t i = 0; i < dims[0]; i++) {
		w[i] = new double* [dims[1]];
		for (size_t j = 0; j < dims[1]; j++) {
			w[i][j] = new double[dims[2]];
			std::fill_n(w[i][j], dims[2], 0.0);
		}

	}

	s.clear();
	for (size_t i = 0; i < dims[0]; i++) {
		double xi1 = i * deltas[0] + xi1_min;
		for (size_t j = 0; j < dims[1]; j++) {
			double theta = theta_min + j * deltas[1];
			int k = 0;
			for (k = 0; xi2_min + deltas[2] * (double)k < 0.0; k++) {
				double xi2 = k * deltas[2] + xi2_min;
				w[i][j][k] = -mu(xi1, xi2) * theta / (1 + theta * theta);
				s.push_back(w[i][j][k]);
			}

			for (k; k < dims[2]; k++) {
				double xi2 = k * deltas[2] + xi2_min;
				w[i][j][k] = mu(xi1, xi2) * theta / (1 + theta * theta);
				s.push_back(w[i][j][k]);
			}
		}
	}

	//std::cout << "w max " << *std::max_element(s.begin(), s.end()) << std::endl;
	double*** v = V(w, u, dims, deltas);
	double*** h = H(u, dims, deltas);

	//std::cout << "-------------" << std::endl;
	// if memory will leak - add delete
	bool stop = false;
	int it_count = 0;

	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
	auto str = oss.str();

	std::stringstream ss;
	const std::string filename = "./out " + str + "/output";

	_mkdir(("./out " + str).c_str());

	//print_array(v, dims);
	print_min_max_values(u, "u", dims);
	print_min_max_values(v, "v", dims);
	print_min_max_values(w, "w", dims);
	print_min_max_values(h, "h", dims);

	ss.str(std::string());
	ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
	export_vector_field(ss.str(), u, v, w, deltas);

	ss.str(std::string());
	ss << filename << "_grid.vts";
	export_grid(ss.str(), deltas);

	std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
	do {
		//std::system("cls");


		double*** h_next = H(h, w, v, dims, deltas, time_step);
		double*** u_next = U(h_next, dims, deltas);
		double*** w_next = W(h_next, w, v, dims, deltas, time_step);
		double*** v_next = V(w_next, u_next, dims, deltas);

		if (it_count++ % print_every == 0) {
			std::cout << "-----------------------------------------------" << std::endl;
			std::cout << " Starting iteration " << it_count << std::endl;
			print_min_max_values(u_next, "u", dims);
			print_min_max_values(v_next, "v", dims);
			print_min_max_values(w_next, "w", dims);
			print_min_max_values(h_next, "h", dims);
		}

		stop = max_norm(u, u_next, dims) < eps && max_norm(v, v_next, dims) < eps && max_norm(w, w_next, dims) < eps;
		double*** old_u = u;
		double*** old_v = v;
		double*** old_w = w;
		double*** old_h = h;

		u = u_next;
		v = v_next;
		w = w_next;
		h = h_next;

		dispose_array(old_u, dims);
		dispose_array(old_v, dims);
		dispose_array(old_w, dims);
		dispose_array(old_h, dims);

		delete[] old_u;
		delete[] old_v;
		delete[] old_w;
		delete[] old_h;

		if (it_count % save_every == 0) {
			ss.str(std::string());
			ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
			export_vector_field(ss.str(), u, v, w, deltas);
		}
	} while (!stop);

	return 0;
}