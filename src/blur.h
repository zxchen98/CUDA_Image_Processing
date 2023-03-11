#ifndef MYCUDA_H
#define MYCUDA_H

void Gaussian_Blur(float* host_input, float* host_output, int nx, int ny, int ksize, float sig);

float* create_filter(double sig, int kernel_size);


#endif