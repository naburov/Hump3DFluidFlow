#pragma once
#include <cmath>


const int N = 100; // number of x points
const int K = 100; // number of y points
const int M = 100; // number of z points
const int T = 100000;

const double x0 = 5.0;
const double z0 = 5.0;
const double eps = 0.0001;
const double f_second = 0.33;
const double c = f_second / std::sqrt(x0);

const double xi1_min = -10;
const double xi1_max = 10;
const double xi2_min = -10;
const double xi2_max = 10;
const double theta_min = 0;
const double theta_max = 10;

const int tmax = 1;

const double alpha = 0.25;
const double beta = 0.25;

const int save_every = 5;
