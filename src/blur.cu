#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "helper_cuda.h"
#include "driver_types.h"
#include <chrono>

#define BLOCK 16
#define FILTER_SIZE 25

using namespace cv;
using namespace std;


/*
    This is the device kernel function that takes in original image and kernel filter to do 2d convolution
*/
__global__ void cuda_convolution(float *input_im, float* output_im, int nx, int ny, float *cuda_kernel, int kernel_size){
    __shared__ float shared_input[FILTER_SIZE* FILTER_SIZE];

    int blockx = blockDim.x;
    int blocky = blockDim.y;

    int idx = blockIdx.x * blockx + threadIdx.x;
    int idy = blockIdx.y * blocky + threadIdx.y;
    int index = idy * nx + idx;

    if(idx>=ny || idy>=nx){
        return;
    }
    else {
        // copy filter value into shared memory
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                shared_input[i * kernel_size + j] = cuda_kernel[i * kernel_size + j];
            }
        }
        __syncthreads();

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                // deal with the thread out of bound cases
                int imgx = min(max(idx + i - kernel_size / 2, 0), nx - 1);
                int imgy = min(max(idy + j - kernel_size / 2, 0), ny - 1);
                // convolve with the gaussian kernel
                output_im[index] += input_im[imgy * nx + imgx] * shared_input[j * kernel_size + i];
            }
        }
    }
}

/*
    This funtion will allocate memory in cuda as well as define block and grid size. It will also measure the device funtion running time
*/
void Gaussian_Blur(float *host_input, float *host_output, int nx, int ny, float* host_kernel, float* cuda_kernel, int filter_size){
    
    // allocate memory for both input and output image
    float *input_im;
    checkCudaErrors(cudaMalloc((void **)&input_im, nx * ny * sizeof(float)));
    checkCudaErrors(cudaMemcpy(input_im, host_input, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    float* output_im;
    checkCudaErrors(cudaMalloc((void**)&output_im, nx * ny * sizeof(float)));

    // create filter and tranfer to device
    checkCudaErrors(cudaMalloc((void **)&cuda_kernel, filter_size * filter_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(cuda_kernel, host_kernel, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

    // run kernel function and measure running time
    dim3 block = dim3(BLOCK, BLOCK, 1);
    dim3 grid = dim3(ceil(nx / block.x) , ceil(ny / block.y), 1);

    auto cbegin = std::chrono::high_resolution_clock::now();
    cuda_convolution << <grid, block, FILTER_SIZE* FILTER_SIZE * sizeof(float) >> > (input_im, output_im, nx, ny, cuda_kernel, filter_size);
    auto cend = std::chrono::high_resolution_clock::now();
    auto ctime = std::chrono::duration_cast<std::chrono::nanoseconds>(cend - cbegin);
    cout << "Processing time:\n" << endl;
    cout << "Processing Time With Cuda: " << ctime.count() * 1e-9 << " sec" << endl;

    checkCudaErrors(cudaMemcpy(host_output, output_im, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
}