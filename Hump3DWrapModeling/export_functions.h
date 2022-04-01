#include <string>
#pragma once
void print_array(double*** arr, const int(&dims)[3]);
void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3]);
void dispose_array(double*** arr, const int(&dims)[3]);
void print_min_max_values(double*** arr, std::string name, const int(&dims)[3]);
void export_grid(const std::string& filename, const double(&deltas)[3]);