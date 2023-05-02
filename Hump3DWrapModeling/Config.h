//
// Created by Буров  Никита on 27.11.2022.
//

#ifndef HUMP3DFLUIDFLOW_CONFIG_H
#define HUMP3DFLUIDFLOW_CONFIG_H

#include <map>
#include <string>
#include <vector>

enum functions {
    tanh_, exp_
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

    int get_max_steps();

    Config() = delete;

    Config(const std::string &config_filename);

    void print();

    std::vector<double> get_hump_center();
};


#endif //HUMP3DFLUIDFLOW_CONFIG_H
