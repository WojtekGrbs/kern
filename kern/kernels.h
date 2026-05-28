#ifndef KERNELS_1D
#define KERNELS_1D

#include <numpy/arrayobject.h>
#include <math.h>

/*
standard kernels, behaving in a regular way
*/
#define SQRT_TWO_PI_INVERSE 0.39894228040143267794 // constant for gaussian kernel

static inline double gaussian_1d_kernel(double x){
    return 0.3989422804014327 * exp(-0.5 * x * x);
}

/*
weird kernels, behaving in their own way
*/
void standard_kde(double* x, npy_intp n, double h, double* out);



#endif