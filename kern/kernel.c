#include "kernels.h"
#include <numpy/arrayobject.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MIN(a, b) ((a)<(b)?(a):(b))

/* 
standard procedure of calculating the kernel denstiy estimation
*/
void standard_kde(double* x, npy_intp n, double h, double* out){

    double u;
    double inv_h = 1/h;
    double self = gaussian_1d_kernel(0.0);
    size_t BLOCKSIZE = 12;
    double norm = 1.0 / ((n-1) * h);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for(size_t i=0; i<n; ++i){
        double xi = x[i];
        double sum=0.0;

        #ifdef _OPENMP
        #pragma omp simd reduction(+:sum)
        #endif
        for(size_t j=0; j<n; ++j){
           sum += gaussian_1d_kernel((xi-x[j])*inv_h);
        }
        sum -= self;
        out[i] = sum * norm;
    }
}