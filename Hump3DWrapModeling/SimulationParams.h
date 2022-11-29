//
// Created by Буров  Никита on 29.11.2022.
//

#ifndef HUMP3DFLUIDFLOW_SIMULATIONPARAMS_H
#define HUMP3DFLUIDFLOW_SIMULATIONPARAMS_H

struct SimulationParams{
    const int dims[3];
    const double mins[3], deltas[3];
    const double alpha, beta, A, timeStep;
};


#endif //HUMP3DFLUIDFLOW_SIMULATIONPARAMS_H
