#include <string>
#include "SimulationParams.h"

#pragma once
void print_array(double*** arr, SimulationParams sim_params);
void export_vector_field(const std::string& filename, double* U, double* V, double* W, SimulationParams sim_params);
void dispose_array(double*** arr, SimulationParams sim_params);
void print_min_max_values(const double *arr, const std::string &name, SimulationParams params) ;
void export_grid(const std::string& filename, SimulationParams sim_params);
void export_central_slice(const std::string& filename, double*** U, double*** V, double*** W, SimulationParams sim_params);
void export_theta_slice(const std::string& filename, double*** U, double*** V, double*** W, SimulationParams sim_params,double theta);
void export_single_line(const std::string& filename, double*** T, double theta, SimulationParams sim_params);
void output_vtk_binary_2d(const std::string& filename, double*** U, double*** V, int k, SimulationParams sim_params);
