#ifndef KDE_H
#define KDE_H

#include "kernels.h"

void self_kde_gaussian(const double *restrict data, double *restrict out,
                       int n, double h);

void self_kde_beta(const double *restrict data, double *restrict out,
                   int n, double h);

void self_kde_generic(const double *restrict data, double *restrict out,
                      int n, double h, const kernel1d_info *kernel);

void ext_kde_gaussian(const double *restrict data, int n,
                      const double *restrict xs, double *restrict out,
                      int m, double h);

void ext_kde_beta(const double *restrict data, int n,
                  const double *restrict xs, double *restrict out,
                  int m, double h);

void ext_kde_generic(const double *restrict data, int n,
                     const double *restrict xs, double *restrict out,
                     int m, double h, const kernel1d_info *kernel);

void self_kde_reflected(const double *restrict data, double *restrict out,
                        int n, double h, const kernel1d_info *kernel);

void ext_kde_reflected(const double *restrict data, int n,
                       const double *restrict xs, double *restrict out,
                       int m, double h, const kernel1d_info *kernel);

#endif
