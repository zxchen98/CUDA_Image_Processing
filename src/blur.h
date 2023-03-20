#ifndef MYCUDA_H
#define MYCUDA_H

void Gaussian_Blur(float* host_input, float* host_output, int nx, int ny,float* host_kernel,float* cuda_kernel,int );


#endif