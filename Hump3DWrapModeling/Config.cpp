//
// Created by Буров  Никита on 27.11.2022.
//

#include <fstream>
#include <iostream>
#include "Config.h"

const char DELIMITER = ':';

void Config::parse_params(const std::string &config_filename) {
    std::ifstream input_stream(config_filename);

    if (!input_stream) std::cout << "Can't open input file!";

    std::string line;
    while (std::getline(input_stream, line)) {
        auto position    = line.find(DELIMITER);
        auto param_name  = line.substr(0, position);
        auto param_value = line.substr(position + 1, line.length() - 1 - param_name.length());
        params.insert({param_name, param_value});
    }
}

int Config::get_max_steps() {
    auto max_steps = std::stoi(params.find("max_steps")->second);
    return max_steps;
}

std::vector<int> Config::get_grid_dimensions() {
    auto N = std::stoi(params.find("N")->second); //x
    auto M = std::stoi(params.find("M")->second); //z
    auto K = std::stoi(params.find("K")->second); //y
    return std::vector<int>({N, K, M});                    //x, y, z
}

std::vector<double> Config::get_grid_sizes() {
    auto xi1_max   = std::stod(params.find("xi1_max")->second);
    auto xi1_min   = std::stod(params.find("xi1_min")->second);
    auto xi2_max   = std::stod(params.find("xi2_max")->second);
    auto xi2_min   = std::stod(params.find("xi2_min")->second);
    auto theta_max = std::stod(params.find("theta_max")->second);
    auto theta_min = std::stod(params.find("theta_min")->second);
    return std::vector<double>({
                                       xi1_max, xi1_min,
                                       xi2_max, xi2_min,
                                       theta_max, theta_min
                               });
}

double Config::get_timestep() {
    auto T    = std::stod(params.find("T")->second);
    auto tmax = std::stod(params.find("tmax")->second);
    return tmax / T;
}

double Config::get_hump_height() {
    return std::stod(params.find("A")->second);;
}

std::vector<double> Config::get_function_params() {
    auto alpha = std::stod(params.find("alpha")->second);
    auto beta  = std::stod(params.find("beta")->second);
    return std::vector<double>({alpha, beta});
}

std::vector<int> Config::get_saving_params() {
    auto save_every  = std::stoi(params.find("save_every")->second);
    auto print_every = std::stoi(params.find("print_every")->second);
    return std::vector<int>({save_every, print_every});
}

std::vector<double> Config::get_hump_center() {
    auto xi1 = std::stod(params.find("x0")->second);
    auto xi2 = std::stod(params.find("z0")->second);
    return std::vector<double>({xi1, xi2});
}

functions Config::get_hump_function() {
    return functions::tanh_;
}

Config::Config(const std::string &config_filename) {
    parse_params(config_filename);
}

void Config::print() {
    for (const auto &pair: params) {
        fprintf(stdout, "%s:%s \n", pair.first.c_str(), pair.second.c_str());
    }
}