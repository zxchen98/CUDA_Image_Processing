#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "blur.h"

using namespace cv;
using namespace std;

Mat cudaGaussian(Mat src, int kernel_size, float sig)
{
    Mat res(src.size(), CV_8UC1);
    int nx = src.cols;
    int ny = src.rows;

    float* host_input = new float[nx * ny];
    float* host_output = new float[ny * nx];
    // y * width + x
    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            host_input[i * nx + j] = uchar(src.at<uchar>(i * nx + j));
        }
    }

    Gaussian_Blur(host_input, host_output, nx, ny, kernel_size, sig);

    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            res.at<uchar>(i, j) = host_output[i * nx + j];
        }
    }
    return res;
}

int main()
{

    string path = "../image/Grab_Image.bmp";
    Mat src = imread(path, 0);
    resize(src, src, Size(4096, 4096));
    int ksize = 45;
    float sigma = 25.0f;

    // CUDA Global Memory Gaussian Filtering
    Mat blurred = cudaGaussian(src, ksize, sigma);


    // Host Gaussian Filtering
    auto s1 = std::chrono::high_resolution_clock::now();
    Mat image_blurred_with_host;
    GaussianBlur(src, image_blurred_with_host, Size(ksize, ksize), 0);
    auto f1 = std::chrono::high_resolution_clock::now();
    float d1 = float(std::chrono::duration_cast<std::chrono::microseconds>(f1 - s1).count()) / 1000000;
    
    cout << "  ---Processing Time With Host         : " << d1 << " sec" << endl;
   
    
    imshow("blurred_image", blurred);

    imwrite("../result/blurred_image.png", blurred);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
