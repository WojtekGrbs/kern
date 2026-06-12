#ifndef KDE_MULTIVARIATE_H
#define KDE_MULTIVARIATE_H

#include "kernels.h"

void multivariate_kde(const double *restrict data, int n, int dimensions,
                      const double *restrict xs, double *restrict out, int m,
                      double h, const kernel1d_info *kernel, int block_size);

void multivariate_self_kde(const double *restrict data, double *restrict out,
                           int n, int dimensions, double h,
                           const kernel1d_info *kernel, int block_size);

#endif
