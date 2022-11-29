#define _CRT_SECURE_NO_WARNINGS

#include <iomanip>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <chrono>

#include "consts.h"
#include "export_functions.h"
#include "calculating_functions.h"
#include "Config.h"
#include "SimulationParams.h"

#include <vector>
#include <algorithm>

#ifdef __APPLE__

#include <sys/stat.h>

#elif _WIN32
#include "direct.h"
#elif __linux__
#include <sys/stat.h>
#endif

int main() {
    omp_set_num_threads(8);

    auto cnf_path = "/Users/burovnikita/CLionProjects/Hump3DFluidFlow/configs/testconf";
    auto cnf = Config(cnf_path);
    cnf.print();

    auto t_dims = cnf.get_grid_dimensions();
    const int dims[] = {t_dims[0], t_dims[1], t_dims[2]};

    auto t_sizes = cnf.get_grid_sizes();


    const double time_step = cnf.get_timestep();

    auto func_params = cnf.get_function_params();

    SimulationParams sim_params = {
            {t_dims[0], t_dims[1], t_dims[2]},
            {t_sizes[1], t_sizes[5], t_sizes[3]},
            {
            (t_sizes[0] - t_sizes[1]) / (t_dims[0]),
            (t_sizes[4] - t_sizes[5]) / (t_dims[1]),
            (t_sizes[2] - t_sizes[3]) / (t_dims[2])
            },
            func_params[0],
            func_params[1],
            cnf.get_hump_height(),
            time_step
    };

    auto t_params = cnf.get_saving_params();
    auto save_every = t_params[0];
    auto print_every = t_params[1];

    // initalize U
    auto ***u = new double **[dims[0]];
    for (size_t i = 0; i < dims[0]; i++) {
        u[i] = new double *[dims[1]];
        for (size_t j = 0; j < dims[1]; j++)
            u[i][j] = new double[dims[2]];

    }

    std::vector<double> s;
    for (size_t i = 0; i < dims[0]; i++)
        for (size_t k = 0; k < dims[2]; k++) {
            auto xi1 = i * sim_params.deltas[0] + t_sizes[1];
            auto xi2 = k * sim_params.deltas[2] + t_sizes[3];
            auto j = 0;
            for (j = 0; t_sizes[5] + sim_params.deltas[1] * (double) j < 5.0; j++) {
                auto theta = t_sizes[5] + j * sim_params.deltas[1];
                u[i][j][k] = f_second * theta * (1 + 0.2 * mu(xi1, xi2, sim_params));
                s.push_back(u[i][j][k]);
            }

            for (j; j < dims[1]; j++) {
                auto theta = t_sizes[5] + j * sim_params.deltas[1];
                u[i][j][k] = f_second * (theta + mu(xi1, xi2, sim_params));
                s.push_back(u[i][j][k]);
            }

        }
    //std::cout << "u max " << *std::max_element(s.begin(), s.end()) << std::endl;

    // initialize W
    auto ***w = new double **[dims[0]];
    for (size_t i = 0; i < dims[0]; i++) {
        w[i] = new double *[dims[1]];
        for (size_t j = 0; j < dims[1]; j++) {
            w[i][j] = new double[dims[2]];
            std::fill_n(w[i][j], dims[2], 0.0);
        }

    }

    s.clear();
    for (size_t i = 0; i < dims[0]; i++) {
        auto xi1 = i * sim_params.deltas[0] + t_sizes[1];
        for (size_t j = 0; j < dims[1]; j++) {
            auto theta = t_sizes[5] + j * sim_params.deltas[1];
            auto k = 0;
            for (k = 0; t_sizes[3] + sim_params.deltas[2] * (double) k < 0.0; k++) {
                auto xi2 = k * sim_params.deltas[2] + t_sizes[3];
                w[i][j][k] = -mu(xi1, xi2, sim_params) * theta / (1 + theta * theta);
                s.push_back(w[i][j][k]);
            }

            for (k; k < dims[2]; k++) {
                auto xi2 = k * sim_params.deltas[2] + t_sizes[3];
                w[i][j][k] = mu(xi1, xi2, sim_params) * theta / (1 + theta * theta);
                s.push_back(w[i][j][k]);
            }
        }
    }

    //std::cout << "w max " << *std::max_element(s.begin(), s.end()) << std::endl;
    auto ***v = V(w, u, sim_params);
    auto ***h = H(u, sim_params);

    //std::cout << "-------------" << std::endl;
    // if memory will leak - add delete
    auto stop = false;
    auto it_count = 0;

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    auto str = oss.str();

    std::stringstream ss;
    const std::string filename = "./out " + str + "/output";

#ifdef __APPLE__
    mkdir(("./out " + str).c_str(), 0777);
#elif _WIN32
    _mkdir(("./out " + str).c_str());
#elif __linux__
    mkdir(("./out " + str).c_str(), 0777);
#endif

    //print_array(v, dims);
    print_min_max_values(u, "u", sim_params);
    print_min_max_values(v, "v", sim_params);
    print_min_max_values(w, "w", sim_params);
    print_min_max_values(h, "h", sim_params);

    ss.str(std::string());
    ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
    export_vector_field(ss.str(), u, v, w, sim_params);

    ss.str(std::string());
    ss << filename << "_grid.vts";
    export_grid(ss.str(), sim_params);

    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
    do {
        //std::system("cls");


        auto ***h_next = H(h, w, v, sim_params);
        auto ***u_next = U(h_next, sim_params);
        auto ***w_next = W(h_next, w, v, sim_params);
        auto ***v_next = V(w_next, u_next, sim_params);

        if (it_count++ % print_every == 0) {
            std::cout << "-----------------------------------------------" << std::endl;
            std::cout << " Starting iteration " << it_count << std::endl;
            print_min_max_values(u_next, "u", sim_params);
            print_min_max_values(v_next, "v", sim_params);
            print_min_max_values(w_next, "w", sim_params);
            print_min_max_values(h_next, "h", sim_params);
        }

        stop = max_norm(u, u_next, sim_params) < eps && max_norm(v, v_next, sim_params) < eps && max_norm(w, w_next, sim_params) < eps;
        auto ***old_u = u;
        auto ***old_v = v;
        auto ***old_w = w;
        auto ***old_h = h;

        u = u_next;
        v = v_next;
        w = w_next;
        h = h_next;

        dispose_array(old_u, sim_params);
        dispose_array(old_v, sim_params);
        dispose_array(old_w, sim_params);
        dispose_array(old_h, sim_params);

        delete[] old_u;
        delete[] old_v;
        delete[] old_w;
        delete[] old_h;

        if (it_count % save_every == 0) {
            ss.str(std::string());
            ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_vector_field(ss.str(), u, v, w,sim_params);

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            ss.str(std::string());
            ss << filename << "_central_slice_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_central_slice(ss.str(), u, v, w, sim_params);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference vtk "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

            ss.str(std::string());
            ss << filename << "_u_theta0_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), u,  0, sim_params);

            ss.str(std::string());
            ss << filename << "_u_theta1_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), u,  1, sim_params);

            ss.str(std::string());
            ss << filename << "_u_theta_max-1_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), u, t_sizes[4] - 1, sim_params);

            ss.str(std::string());
            ss << filename << "_u_theta_max_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), u,  t_sizes[4] - sim_params.deltas[1], sim_params);

            ss.str(std::string());
            ss << filename << "_v_theta0_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), v, 0, sim_params);

            ss.str(std::string());
            ss << filename << "_v_theta1_" << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_single_line(ss.str(), v, 1, sim_params);

            begin = std::chrono::steady_clock::now();
            ss.str(std::string());
            ss << filename << "_bin_vector_field" << std::setfill('0') << std::setw(5) << it_count << ".vtk";
            output_vtk_binary_2d(ss.str(), u, v, it_count, sim_params);
            end = std::chrono::steady_clock::now();
            std::cout << "Time difference binary "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
        }
    } while (!stop);

    dispose_array(u, sim_params);
    dispose_array(v, sim_params);
    dispose_array(w, sim_params);
    dispose_array(h, sim_params);

    delete[] u;
    delete[] v;
    delete[] w;
    delete[] h;

    return 0;
}