#include <string>
#pragma once
void print_array(double*** arr, const int(&dims)[3]);
void export_vector_field(const std::string& filename, double*** U, double*** V, double*** W, const double(&deltas)[3]);
void dispose_array(double*** arr, const int(&dims)[3]);