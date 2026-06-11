#include "bandwidth.h"
#include "kde.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

double loo_score(const double *restrict data, int n, double h, int metric) {
    double *loo_dens = (double *)malloc(n * sizeof(double));
    if (!loo_dens) return -DBL_MAX;

    self_kde_gaussian(data, loo_dens, n, h);

    double score = 0.0;
    if (metric == 0) {
        for (int i = 0; i < n; i++) {
            double d = loo_dens[i];
            score += (d > 1e-300) ? log(d) : -700.0;
        }
        score /= n;
    } else {
        double cross = 0.0;
        for (int i = 0; i < n; i++) {
            cross += loo_dens[i];
        }
        score = cross / n;
    }

    free(loo_dens);
    return score;
}

double kfold_score(const double* restrict data, int n,
                   int k_folds, double h, int metric) {
    double total_score = 0.0;
    int total_count = 0;

    for (int f = 0; f < k_folds; f++) {
        int test_start = (f * n) / k_folds;
        int test_end   = ((f + 1) * n) / k_folds;
        int n_test     = test_end - test_start;
        int n_train    = n - n_test;

        double *train = (double *)malloc(n_train * sizeof(double));
        double *test  = (double *)malloc(n_test  * sizeof(double));
        double *pout  = (double *)malloc(n_test  * sizeof(double));

        if (!train || !test || !pout) {
            free(train);
            free(test);
            free(pout);
            return -DBL_MAX;
        }

        int ti = 0;
        int ri = 0;
        for (int i = 0; i < n; i++) {
            if (i >= test_start && i < test_end) {
                test[ti++] = data[i];
            } else {
                train[ri++] = data[i];
            }
        }

        ext_kde_gaussian(train, n_train, test, pout, n_test, h);

        if (metric == 0) {
            for (int i = 0; i < n_test; i++) {
                double d = pout[i];
                total_score += (d > 1e-300) ? log(d) : -700.0;
            }
        } else {
            for (int i = 0; i < n_test; i++) {
                total_score += pout[i];
            }
        }

        total_count += n_test;

        free(train);
        free(test);
        free(pout);
    }

    return (total_count > 0) ? total_score / total_count : -DBL_MAX;
}
