//
// Created by Буров  Никита on 27.11.2022.
//

#include <fstream>
#include <iostream>
#include "Config.h"

const char DELIMITER = ':';

void Config::parse_params(const std::string &config_filename) {
    std::ifstream input_stream(config_filename);

    if (!input_stream) std::cerr << "Can't open input file!";

    std::string line;
    while (getline(input_stream, line)) {
        auto position = line.find(DELIMITER);
        auto param_name = line.substr(0, position);
        auto param_value = line.substr(position + 1, line.length() - 1 - param_name.length());
        params.insert({param_name, param_value});
    }
}

std::vector<int> Config::get_grid_dimensions() {
    return std::vector<int>();
}

std::vector<double> Config::get_grid_sizes() {
    return std::vector<double>();
}

double Config::get_timestep() {
    auto T = std::stod(params.find("T")->second);
    auto tmax = std::stod(params.find("tmax")->second);
    return tmax / T;
}

double Config::get_hump_height() {
    return std::stod(params.find("A")->second);;
}

std::vector<double> Config::get_function_params() {
    return std::vector<double>();
}

std::vector<int> Config::get_saving_params() {
    return std::vector<int>();
}

functions Config::get_hump_function() {
    return functions::tanh;
}

Config::Config(const std::string &config_filename) {
    parse_params(config_filename);
}

void Config::print() {
    for (const auto &pair: params) {
        std::cout << pair.first << ":" << pair.second << std::endl;
    }
}