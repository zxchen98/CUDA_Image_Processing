#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "driver_types.h"
#include "helper_cuda.h"
#include <chrono>

using namespace cv;
using namespace std;


__global__ void cuda_process(float *input_im, float* output_im, int nx, int ny, float *cuda_kernel, int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * nx + idx;

    if(idx>=ny || idy>=nx){
        return;
    }

    for (int i = 0; i < kernel_size; i++){
        for (int j = 0; j < kernel_size; j++){
            // deal with the pixel out of bound cases
            int imgx = min(max(idx + i - kernel_size / 2, 0), nx - 1);
            int imgy = min(max(idy + j - kernel_size / 2, 0), ny - 1);
            // convolve with the gaussian kernel
            output_im[index] += input_im[imgy * nx + imgx] * cuda_kernel[j * kernel_size + i];
        }
    }
}

float* create_filter(double sig, int kernel_size)
{
    float* kernel = new float[kernel_size * kernel_size];
    float total_sum = 0.0;
    // generate the gaussian filter
    for (int i = 0; i < kernel_size; i++){
        for (int j = 0; j < kernel_size; j++) {
            float c = float(j - kernel_size / 2);
            float r = float(i - kernel_size / 2);
            kernel[j * kernel_size + i] = (float)exp(-(c*c + r*r)/(2*sig*sig)) / (2.0*sig*sig);
            total_sum += kernel[j * kernel_size + i];
        }
        }
    // apply normalization factor
    float normalization_factor = 1.0 / total_sum;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[j * kernel_size + i] *= normalization_factor;
            }
        }
    return kernel;
    }

void Gaussian_Blur(float *host_input, float *host_output, int nx, int ny, int ksize, float sig)
{
    // allocate memory for both input and output image
    float *input_im;
    float *output_im;
    checkCudaErrors(cudaMalloc((void **)&input_im, nx * ny * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output_im, nx * ny * sizeof(float)));
    checkCudaErrors(cudaMemcpy(input_im, host_input, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    float *cuda_kernel = new float[ksize * ksize];
    float* host_kernel;
    double sigma = sig;

    // create filter and tranfer to device
    host_kernel = create_filter(sigma, ksize);
    checkCudaErrors(cudaMalloc((void **)&cuda_kernel, ksize * ksize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(cuda_kernel, host_kernel, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

    // run kernel function and measure running time
    dim3 block = dim3(16, 16, 1);
    dim3 grid = dim3(nx / block.x + 1, ny / block.y + 1, 1);

    auto s0 = std::chrono::high_resolution_clock::now();
    cuda_process << <grid, block,sizeof(float)*ksize*ksize >> > (input_im, output_im,nx, ny, cuda_kernel, ksize);
    auto f0 = std::chrono::high_resolution_clock::now();
    float d0 = float(std::chrono::duration_cast<std::chrono::microseconds>(f0 - s0).count()) / 1000000;

    cout << "  --- Processing Time..." << endl;
    cout << "  ---Processing Time With Cuda         : " << d0 << " sec" << endl;

    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaMemcpy(host_output, output_im, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
}