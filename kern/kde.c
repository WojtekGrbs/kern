#include "kde.h"

#if defined(_OPENMP)
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

    /* self-KDE with Gaussian kernel
    * out[i] = (1/((n-1)*h)) * sum_{j!=i} K_gauss((xi - xj)/h)
    */
void self_kde_gaussian(const double *restrict data, double *restrict out,
                       int n, double h) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))
    for (int i = 0; i < n; i++) {
        double xi = data[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < i; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = i + 1; j < n; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        out[i] = sum * inv_nh;
    }
}

void self_kde_beta(const double *restrict data, double *restrict out,
                   int n, double h) {
    double inv_nm1 = 1.0 / (n - 1);

    OMP_PRAGMA(omp parallel for schedule(dynamic, 32))
    for (int i = 0; i < n; i++) {
        double xi  = data[i];
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            sum += kernel_beta(xi, data[j], h);
        }

        out[i] = sum * inv_nm1;
    }
}

void self_kde_generic(const double *restrict data, double *restrict out,
                      int n, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(dynamic))
    for (int i = 0; i < n; i++) {
        double xi = data[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double diff = xi - data[j];
                sum += kfn(diff * inv_h);
            }
        }

        out[i] = sum * inv_nh;
    }
}

void ext_kde_gaussian(const double *restrict data, int n,
                      const double *restrict xs, double *restrict out,
                      int m, double h) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi = xs[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < n; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        out[i] = sum * inv_nh;
    }
}

void ext_kde_beta(const double *restrict data, int n,
                  const double *restrict xs, double *restrict out,
                  int m, double h) {
    double inv_n = 1.0 / n;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < n; j++) {
            sum += kernel_beta(xi, data[j], h);
        }

        out[i] = sum * inv_n;
    }
}

void ext_kde_generic(const double *restrict data, int n,
                     const double *restrict xs, double *restrict out,
                     int m, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < n; j++) {
            double diff = xi - data[j];
            sum += kfn(diff * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

void self_kde_reflected(const double *restrict data, double *restrict out,
                        int n, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))
    for (int i = 0; i < n; i++) {
        double xi  = data[i];
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;

            double xj = data[j];
            sum += kfn((xi - xj) * inv_h);
            sum += kfn((xi + xj) * inv_h);
            sum += kfn((xi - (2.0 - xj)) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

/* Boundary-reflect KDE, all points contribute. */
void ext_kde_reflected(const double *restrict data, int n,
                       const double *restrict xs, double *restrict out,
                       int m, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;

        OMP_PRAGMA(omp simd reduction(+:sum))
        for (int j = 0; j < n; j++) {
            double xj = data[j];
            sum += kfn((xi - xj) * inv_h);
            sum += kfn((xi + xj) * inv_h);
            sum += kfn((xi - (2.0 - xj)) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}
