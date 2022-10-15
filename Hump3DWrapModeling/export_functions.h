#include <string>
#pragma once
void print_array(double*** arr, const int(&dims)[3]);
void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3]);
void dispose_array(double*** arr, const int(&dims)[3]);
void print_min_max_values(double*** arr, std::string name, const int(&dims)[3]);
void export_grid(const std::string& filename, const double(&deltas)[3]);
void export_central_slice(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3]);
void export_theta_slice(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3], double theta);
void export_single_line(const std::string& filename, double*** T, const double(&deltas)[3], double theta);
void output_vtk_binary_2d(const std::string& filename, double*** U, double*** V, int k, const double(&deltas)[3]);