#ifndef KDE_BANDWIDTH_H
#define KDE_BANDWIDTH_H

#include "kernels.h"

enum {
    BANDWIDTH_PARALLEL_AUTO = 0,
    BANDWIDTH_PARALLEL_GRID = 1,
    BANDWIDTH_PARALLEL_EVALUATION = 2
};

double loo_score_kernel(const double *restrict data, int n, double h,
                        const kernel1d_info *kernel);

double kfold_score_kernel(const double *restrict data, int n,
                          int k_folds, double h,
                          const kernel1d_info *kernel);

void bandwidth_score_grid(const double *restrict data, int n,
                          const double *restrict h_grid, int n_bandwidths,
                          const kernel1d_info *kernel, int k_folds,
                          int parallel_mode,
                          double *restrict scores);

double loo_score(const double *restrict data, int n, double h);
double kfold_score(const double *restrict data, int n,
                   int k_folds, double h);

#endif
