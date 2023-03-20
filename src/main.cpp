#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "blur.h"
#include<direct.h>

using namespace cv;
using namespace std;

// edit them to achieve your desired blurring effect
#define SIGMA 15.0
#define KERNEL_SIZE 55

/*
    This function creates a gaussian filter kernel of defined size

    INPUT: None
    OUTPUT: float pointer to the filter
*/
float* create_filter(){
    // uses the MACRO std and kernel size
    int kernel_size = KERNEL_SIZE * KERNEL_SIZE;
    float* kernel = new float[kernel_size];
    double total_sum = 0;

    // generate the gaussian filter
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {

            float x = float(j - KERNEL_SIZE / 2 - 1);
            float y = float(i - KERNEL_SIZE / 2 - 1);

            int k_idx = j * KERNEL_SIZE + i;
            kernel[k_idx] = 1/(exp((float)(x * x + y * y) / (2 * SIGMA * SIGMA)) / (2.0 * 3.14 * SIGMA * SIGMA));
            total_sum += kernel[k_idx];
        }
    }
    // apply normalization factor
    float normalization_factor = 1.0 / total_sum;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            int k_idx = j * KERNEL_SIZE + i;
            kernel[k_idx] *= normalization_factor;
        }
    }
    //float* host_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    //float* cuda_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    //checkCudaErrors(cudaMalloc((void**)&cuda_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    //checkCudaErrors(cudaMemcpy(cuda_kernel, host_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    return kernel;
}
/*
    This function preprocess the image by converting MAT to float arr
    and then run the cuda processing funtion

    INPUT: MAT image 
    OUTPUT: pointer to the blurred image
*/
float* process_cuda(Mat src){

    int img_size = src.rows * src.cols;

    float* host_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    float* cuda_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];

    // preprocess convert MAT to float
    float* host_input = new float[img_size];
    float* host_output = new float[img_size];


    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            host_input[i * src.cols + j] = uchar(src.at<uchar>(i * src.cols + j));
        }
    }
    // call create_filter to generate gaussian filter
    host_kernel = create_filter();
    // call blurring funtion to process image blur
    Gaussian_Blur(host_input, host_output, src.cols, src.rows, host_kernel,cuda_kernel, KERNEL_SIZE);
    
    return host_output;
}


int main(){

    string path = "../input/Grab_Image.bmp";
    int filter_size = KERNEL_SIZE;

    Mat src = imread(path, 0); //only the gray scale
    if (!src.data || !src.isContinuous()) {
        cout << "The image file is invalid" << endl;
        exit(EXIT_FAILURE);
    }

    if (filter_size * 2 >= src.cols || filter_size * 2 >= src.rows) {
        cout << "please adjust your filter size" << endl;
        exit(EXIT_FAILURE);
    }
    //##################################################################################
    // CUDA Global Memory Gaussian Filtering
    float* blurred = process_cuda(src);
    // convert back to MAT
    Mat res(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            res.at<uchar>(i, j) = blurred[i * src.cols + j];
        }
    }
    //##################################################################################
    // Host Gaussian Filtering
    auto begin = std::chrono::high_resolution_clock::now();
    Mat image_blurred_with_host;
    GaussianBlur(src, image_blurred_with_host, Size(filter_size, filter_size), 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    cout << "Processing Time With Host: " << time.count() * 1e-9 << " sec" << endl;
   
    
    imshow("blurred_image", res);

    if (mkdir("../output") == -1)
        cerr << " Error : " << strerror(errno) << endl;

    else
        cout << "output folder Created" << endl;

    imwrite("../output/blurred_image.png", res);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
