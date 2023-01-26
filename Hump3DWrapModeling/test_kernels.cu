//
// Created by Буров  Никита on 16.01.2023.
//
//#include <cuda.h>
#include <stdio.h>
#include<stdlib.h>
#include <algorithm>

#include "cell_calculating_functions.cuh"
#include "SimulationParams.h"
#include "3D_stencil.cuh"

//int _main(int argc, char **argv) {
//    double arr[27]     = {0, 0, 0, 0, 4., 0, 0, 0, 0,
//                          0, 1, 0, 1, 2, 1, 0, 1, 0,
//                          0, 0, 0, 0, 1, 0, 0, 0, 0};
//    double max_arr[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//    double v_array[64] = {};
//    std::fill_n(v_array, 64, 1.);
//
//    auto   ptr       = &arr;
//    auto   max_ptr   = &max_arr;
//    auto   v_ptr     = &v_array;
//    auto   bytes     = 64 * sizeof(double);
//    auto   max_bytes = 11 * sizeof(double);
//    double *gpu_arr, *out, *max_gpu_arr, *d_v;
//
//    auto out_cpu = (double *) malloc(2 * sizeof(double));
//
//    SimulationParams sim_params = {
//            {3, 3, 3},
//            {0, 0, 0},
//            {1, 1, 1},
//            1,
//            1,
//            1,
//            0.1
//    };
//
//
//    SimulationParams v_sim_params = {
//            {4, 4, 4},
//            {0, 0, 0},
//            {1, 1, 1},
//            1,
//            1,
//            1,
//            0.1
//    };
//
//    SimulationParams *d_sim_params, *d_v_sim_params;
//
//    auto res     = cudaMalloc(&gpu_arr, bytes);
//    auto max_res = cudaMalloc(&max_gpu_arr, max_bytes);
//    if (res != cudaSuccess) {
//        std::cout << "Error occured " << res << std::endl;
//    }
//    cudaMalloc(&out, 2 * sizeof(double));
//    cudaMalloc(&d_sim_params, sizeof(SimulationParams));
//    cudaMalloc(&d_v_sim_params, sizeof(SimulationParams));
//    cudaMalloc(&d_v, 64 * sizeof(double));
//
//    cudaMemcpy(gpu_arr, ptr, bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_v, v_ptr, bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(max_gpu_arr, max_ptr, max_bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_sim_params, &sim_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_v_sim_params, &v_sim_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
//
//    std::cout << "Executing kernel" << std::endl;
//    test_stencil_kernel<<< 1, 1 >>>(gpu_arr, out, 1, 1, 1, d_sim_params);
//    test_index_kernel<<<1, 1>>>(d_sim_params);
//    test_index_back_transform<<<1, 1>>>(d_sim_params);
//    integrate_v_kernel<<<2, 2>>>(d_v, d_v_sim_params);
////    reduce_max_kernel<<<4, 4>>>(max_gpu_arr, 11);
////    main_kernel<<<4, 4>>>();
////    cudaDeviceSynchronize();
//    std::cout << "Kernel finished" << std::endl;
//
//    cudaMemcpy(out_cpu, out, 2 * sizeof(double), cudaMemcpyDeviceToHost);
//    cudaMemcpy(v_array, d_v, 64 * sizeof(double), cudaMemcpyDeviceToHost);
//    std::cout << "Transfer back is done" << std::endl;
//    for (int i = 0; i < 64; i++) {
//        auto id = indexof(i, &v_sim_params);
//        std::cout << id.x << " " << id.y << " " << id.z << " " << v_array[i] << std::endl;
//    }
//    std::cout << std::endl;
//    std::cout << "Print of values is done" << std::endl;
//
//    cudaFree(gpu_arr);
//    cudaFree(out);
//
//    return 0;
//}

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
