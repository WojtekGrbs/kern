#ifndef KDE_APPROX_H
#define KDE_APPROX_H

#include "kernels.h"

enum {
    KDE_APPROX_MEMORY_LOW = 0,
    KDE_APPROX_MEMORY_HIGH = 1,
    KDE_APPROX_MEMORY_AUTO = 2
};

void approx_self_kde(const double *restrict sorted_data,
                     double *restrict out,
                     int n, double h,
                     const kernel1d_info *kernel,
                     double cutoff,
                     int max_neighbors,
                     int fast_gaussian,
                     int memory_mode);

void approx_ext_kde(const double *restrict sorted_data, int n,
                    const double *restrict xs,
                    double *restrict out, int m, double h,
                    const kernel1d_info *kernel,
                    double cutoff,
                    int max_neighbors,
                    int fast_gaussian,
                    int memory_mode);

#endif
