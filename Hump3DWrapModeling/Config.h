//
// Created by Буров  Никита on 27.11.2022.
//

#ifndef HUMP3DFLUIDFLOW_CONFIG_H
#define HUMP3DFLUIDFLOW_CONFIG_H

#include <map>
#include <string>

enum functions {
    tanh, exp
};

class Config {
private:
    std::map<std::string, std::string> params;

    void parse_params(const std::string &config_filename);

public:
    std::vector<int> get_grid_dimensions();

    std::vector<double> get_grid_sizes();

    double get_timestep();

    double get_hump_height();

    std::vector<double> get_function_params();

    std::vector<int> get_saving_params();

    functions get_hump_function();

    Config() = delete;

    Config(const std::string &config_filename);

    void print();
};


#endif //HUMP3DFLUIDFLOW_CONFIG_H
