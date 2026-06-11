#ifndef KDE_BANDWIDTH_H
#define KDE_BANDWIDTH_H

double loo_score(const double *restrict data, int n, double h, int metric);

double kfold_score(const double *restrict data, int n,
                   int k_folds, double h, int metric);

#endif // KDE_BANDWIDTH_H
