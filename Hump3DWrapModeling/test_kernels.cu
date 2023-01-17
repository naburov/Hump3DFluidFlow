//
// Created by Буров  Никита on 16.01.2023.
//
//#include <cuda.h>
#include <stdio.h>
#include<stdlib.h>

#include "cell_calculating_functions.cuh"
#include "SimulationParams.h"
#include "3D_stencil.cuh"

__global__ void test_stencil_kernel(const double* arr, double* out, int i, int j, int k, SimulationParams* params){
    auto stencil = Stencil3D(arr, i, j, k, params);
    stencil.print_stencil();
    out[0] = stencil.x_more.w;
    out[1] = stencil.x_more.w;
}

__global__ void test_index_back_transform(SimulationParams* params){
    printf("0: ");
    auto dim_id = indexof(0, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("26: ");
    dim_id = indexof(26, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("9: ");
    dim_id = indexof(9, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("3: ");
    dim_id = indexof(3, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("1: ");
    dim_id = indexof(1, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("12: ");
    dim_id = indexof(12, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

    printf("14: ");
    dim_id = indexof(14, params);
    printf("[%d %d %d] \n", dim_id.x, dim_id.y, dim_id.z);

}


__global__ void test_index_kernel(SimulationParams* params){
    printf("[2, 2, 2]: ");
    auto flat_id = indexof(2, 2, 2, params);
    printf("%d \n", flat_id);

    printf("[0, 0, 0]: ");
    flat_id = indexof(0, 0, 0, params);
    printf("%d \n", flat_id);

    printf("[1, 0, 0]: ");
    flat_id = indexof(1, 0, 0, params);
    printf("%d \n", flat_id);

    printf("[0, 1, 0]: ");
    flat_id = indexof(0, 1, 0, params);
    printf("%d \n", flat_id);

    printf("[0, 0, 1]: ");
    flat_id = indexof(0, 0, 1, params);
    printf("%d \n", flat_id);

    printf("[1, 1, 0]: ");
    flat_id = indexof(1, 1, 0, params);
    printf("%d \n", flat_id);

    printf("[1, 1, 2]: ");
    flat_id = indexof(1, 1, 2, params);
    printf("%d \n", flat_id);
}
