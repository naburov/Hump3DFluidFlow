#define _CRT_SECURE_NO_WARNINGS

void process_one_config(const char *cnf_path);

void process_one_config_cuda(const char *cnf_path);

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <fstream>

#include "export_functions.h"
#include "Config.h"
#include "SimulationParams.h"

#include <vector>
#include <algorithm>
#include<stdio.h>
#include<stdlib.h>

#ifdef __APPLE__

#include <sys/stat.h>

#elif _WIN32
#include "direct.h"
#elif __linux__

#include <sys/stat.h>

#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include "test_kernels.cuh"
#include "cell_calculating_functions.cuh"
#include "calculating_kernels.cuh"

// TODO: graph execution
// TODO: time becnchmarks - done
// TODO: dp/dxi
// TODO: +/- in H_kernel
// TODO: reduce_max_kernel

void filecopy(FILE *dest, FILE *src) {
    const int size = 16384;
    char      buffer[size];

    while (!feof(src)) {
        int n = fread(buffer, 1, size, src);
        fwrite(buffer, 1, n, dest);
    }

    fflush(dest);
}

int main(int argc, char *argv[]) {

    std::vector<std::string> cnfs;
    for (int                 i = 1; i < argc; ++i) {
        cnfs.emplace_back(argv[i]);
    }

    // /tmp/tmp.SJNETVCrGC/cmake-build-release-x86_64-llvm-homewsl/Hump3DFluidFlow
    // /tmp/tmp.SJNETVCrGC/configs/testconf

    for (const auto &cnf: cnfs) {
        std::cout << "Processing cnf " << cnf << std::endl;
        process_one_config_cuda(cnf.c_str());
    }
    std::cout << "Done processing";
    return 0;
}

void process_one_config_cuda(const char *cnf_path) {
    //region ConfigParsing
    auto cnf = Config(cnf_path);
    cnf.print();

    auto      t_dims = cnf.get_grid_dimensions();
    const int dims[] = {t_dims[0], t_dims[1], t_dims[2]};

    auto t_sizes = cnf.get_grid_sizes();


    const double time_step = cnf.get_timestep();

    auto func_params = cnf.get_function_params();

    SimulationParams sim_params = {
            {t_dims[0], t_dims[1], t_dims[2]},
            {t_sizes[1], t_sizes[5], t_sizes[3]},
            {
                    (t_sizes[0] - t_sizes[1]) / (t_dims[0]),
                    (t_sizes[4] - t_sizes[5]) / (t_dims[1]),
                    (t_sizes[2] - t_sizes[3]) / (t_dims[2])
            },
            func_params[0],
            func_params[1],
            cnf.get_hump_height(),
            time_step
    };

    auto t_params    = cnf.get_saving_params();
    auto save_every  = t_params[0];
    auto print_every = t_params[1];
    auto max_steps   = cnf.get_max_steps();
    //endregion ConfigParsing

    printf("Config is parsed\n");

    //region FileManagementInit
    auto t  = time(nullptr);
    auto tm = *localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    auto str = oss.str();

    std::stringstream ss;
    const std::string filename = "./out " + str + "/output";

#ifdef __APPLE__
    mkdir(("./out " + str).c_str(), 0777);
#elif _WIN32
    _mkdir(("./out " + str).c_str());
#elif __linux__
    mkdir(("./out " + str).c_str(), 0777);
#endif

    ss.str(std::string());
    ss << filename << "-cnf";

    FILE *infile  = fopen(cnf_path, "rb");
    FILE *outfile = fopen(ss.str().c_str(), "wb");

    filecopy(outfile, infile);

    fclose(infile);
    fclose(outfile);

    ss.str(std::string());
    ss << filename << "_grid.vts";
    export_grid(ss.str(), sim_params);
    //endregion FileManagementInit

    printf("Folders are created\n");

    //region cpu tensors allocation
    size_t grid_size_bytes = t_dims[0] * t_dims[1] * t_dims[2] * sizeof(double);
    auto   U               = (double *) malloc(grid_size_bytes);
    auto   H               = (double *) malloc(grid_size_bytes);
    auto   W               = (double *) malloc(grid_size_bytes);
    auto   V               = (double *) malloc(grid_size_bytes);
    //endregion U and W initialization

    //region CudaMemoryInitTransfer
    auto             num_threads_per_block = 512;
    auto             num_blocks            = static_cast<int>(std::ceil(
            static_cast<double>(t_dims[0] * t_dims[1] * t_dims[2]) / num_threads_per_block));
    double           *d_U, *d_H, *d_W, *d_V,
                     *d_old_U, *d_old_H, *d_old_W, *d_old_V;
    SimulationParams *d_sim_params;
    cudaMalloc(&d_H, grid_size_bytes);
    cudaMalloc(&d_U, grid_size_bytes);
    cudaMalloc(&d_W, grid_size_bytes);
    cudaMalloc(&d_V, grid_size_bytes);

    cudaMalloc(&d_old_H, grid_size_bytes);
    cudaMalloc(&d_old_U, grid_size_bytes);
    cudaMalloc(&d_old_W, grid_size_bytes);
    cudaMalloc(&d_old_V, grid_size_bytes);

    cudaMalloc(&d_sim_params, sizeof(SimulationParams));

    cudaMemcpy(d_sim_params, &sim_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, H, grid_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, grid_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, grid_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, grid_size_bytes, cudaMemcpyHostToDevice);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stdout, "Error occured while memorytransfer, %s \n", cudaGetErrorString(err));
    }
    //endregion CudaMemoryTransfer

    //region initial conditions
    u_init_kernel<<<num_blocks, num_threads_per_block>>>(d_U, d_sim_params);
    err = cudaGetLastError();

    w_init_kernel<<<num_blocks, num_threads_per_block>>>(d_W, d_sim_params);
    err = cudaGetLastError();

    h_kernel<<<num_blocks, num_threads_per_block>>>(d_U, d_H, d_sim_params);
    err = cudaGetLastError();

    v_func_kernel<<<num_blocks, num_threads_per_block>>>(d_W, d_V, d_U, d_sim_params);
    err = cudaGetLastError();

    integrate_v_kernel<<<num_blocks, num_threads_per_block>>>(d_V, d_sim_params);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stdout, "Error occured, %s \n", cudaGetErrorString(err));
    }
    //endregion initial conditions

    cudaDeviceSynchronize();

    auto stop     = false;
    auto it_count = 0;

    cudaEvent_t kernel_event_start, kernel_event_stop, it_start, it_stop;

    std::vector<float> h_times;
    std::vector<float> u_times;
    std::vector<float> w_times;
    std::vector<float> v_times;
    std::vector<float> it_times;

    cudaEventCreate(&kernel_event_start);
    cudaEventCreate(&kernel_event_stop);
    cudaEventCreate(&it_start);
    cudaEventCreate(&it_stop);

    while (!stop) {
        stop = it_count > max_steps;
        //region PointersReassign
        auto temp_h = d_old_H;
        auto temp_u = d_old_U;
        auto temp_w = d_old_W;
        auto temp_v = d_old_V;

        d_old_H = d_H;
        d_old_U = d_U;
        d_old_W = d_W;
        d_old_V = d_V;

        d_H = temp_h;
        d_U = temp_u;
        d_W = temp_w;
        d_V = temp_v;
        //endregion PointersReassign
        cudaEventRecord(it_start);

        cudaEventRecord(kernel_event_start);
        h_kernel<<<num_blocks, num_threads_per_block>>>(d_old_H, d_old_W, d_old_V, d_H, d_sim_params);
        cudaEventRecord(kernel_event_stop);
        cudaEventSynchronize(kernel_event_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, kernel_event_start, kernel_event_stop);
        h_times.push_back(milliseconds);

        cudaEventRecord(kernel_event_start);
        u_kernel<<<num_blocks, num_threads_per_block>>>(d_H, d_U, d_sim_params);
        cudaEventRecord(kernel_event_stop);
        cudaEventSynchronize(kernel_event_stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, kernel_event_start, kernel_event_stop);
        u_times.push_back(milliseconds);

        cudaEventRecord(kernel_event_start);
        w_kernel<<<num_blocks, num_threads_per_block>>>(d_H, d_old_W, d_old_V, d_W, d_sim_params);
        cudaEventRecord(kernel_event_stop);
        cudaEventSynchronize(kernel_event_stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, kernel_event_start, kernel_event_stop);
        w_times.push_back(milliseconds);
        cudaDeviceSynchronize();

        cudaEventRecord(kernel_event_start);
        v_func_kernel<<<num_blocks, num_threads_per_block>>>(d_W, d_V, d_U, d_sim_params);
        integrate_v_kernel<<<num_blocks, num_threads_per_block>>>(d_V, d_sim_params);
        cudaEventRecord(kernel_event_stop);
        cudaEventSynchronize(kernel_event_stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, kernel_event_start, kernel_event_stop);
        v_times.push_back(milliseconds);

        cudaEventRecord(it_stop);
        cudaEventSynchronize(it_stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, it_start, it_stop);
        it_times.push_back(milliseconds);

//        std::cout << "-----------------------------------------------" << std::endl;
//        std::cout << " Starting iteration " << it_count++ << std::endl;
//        cudaMemcpy(H, d_old_H, grid_size_bytes, cudaMemcpyDeviceToHost);
//        cudaMemcpy(U, d_old_U, grid_size_bytes, cudaMemcpyDeviceToHost);
//        cudaMemcpy(W, d_old_W, grid_size_bytes, cudaMemcpyDeviceToHost);
//        cudaMemcpy(V, d_old_V, grid_size_bytes, cudaMemcpyDeviceToHost);
//        for(int i = 0; i < sim_params.dims[0] * sim_params.dims[1] * sim_params.dims[2]; i++){
//            auto id = indexof(i, &sim_params);
//            std::cout << "x: " << id.x << " y: " << id.y << " z: " << id.z << " v: " << H[i] << std::endl;
//        }
//        if (it_count == 3)
//            return
//        print_min_max_values(U, "u", sim_params);
//        print_min_max_values(V, "v", sim_params);
//        print_min_max_values(W, "w", sim_params);
//        print_min_max_values(H, "h", sim_params);

//        ss.str(std::string());
//        ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
//        export_vector_field(ss.str(), U, V, W, sim_params);
//        it_count++;

        if (it_count++ % print_every == 0) {
            std::cout << "-----------------------------------------------" << std::endl;
            std::cout << " Starting iteration " << it_count << std::endl;
            std::cout << "Average h_kernel: " << std::accumulate(h_times.begin(), h_times.end(), 0.0) / h_times.size()
                      << std::endl;
            std::cout << "Average u_kernel: " << std::accumulate(u_times.begin(), u_times.end(), 0.0) / u_times.size()
                      << std::endl;
            std::cout << "Average w_kernel: " << std::accumulate(w_times.begin(), w_times.end(), 0.0) / w_times.size()
                      << std::endl;
            std::cout << "Average v_kernel: " << std::accumulate(v_times.begin(), v_times.end(), 0.0) / v_times.size()
                      << std::endl;
            std::cout << "Average it time: " << std::accumulate(it_times.begin(), it_times.end(), 0.0) / it_times.size()
                      << std::endl;
        }

        if (it_count % save_every == 0) {
            cudaMemcpy(H, d_old_H, grid_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(U, d_old_U, grid_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(W, d_old_W, grid_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(V, d_old_V, grid_size_bytes, cudaMemcpyDeviceToHost);

            print_min_max_values(U, "u", sim_params);
            print_min_max_values(V, "v", sim_params);
            print_min_max_values(W, "w", sim_params);
            print_min_max_values(H, "h", sim_params);

            ss.str(std::string());
            ss << filename << std::setfill('0') << std::setw(5) << it_count << ".vts";
            export_vector_field(ss.str(), U, V, W, sim_params);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stdout, "Error occured, %s \n", cudaGetErrorString(err));
        }

    }

    cudaFree(d_H);
    cudaFree(d_U);
    cudaFree(d_W);
    cudaFree(d_V);
    cudaFree(d_old_H);
    cudaFree(d_old_U);
    cudaFree(d_old_W);
    cudaFree(d_old_V);
    cudaFree(d_sim_params);

    delete[] U;
    delete[] V;
    delete[] W;
    delete[] H;

}