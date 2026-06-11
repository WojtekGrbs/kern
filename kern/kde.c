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

#define DEFINE_SELF_KDE_GENERIC(NAME, KFN)                                      \
static void self_kde_generic_##NAME(const double *restrict data,                \
                                    double *restrict out,                       \
                                    int n, double h) {                          \
    double inv_nh = 1.0 / ((n - 1) * h);                                        \
    double inv_h  = 1.0 / h;                                                    \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))                          \
    for (int i = 0; i < n; i++) {                                               \
        double xi = data[i];                                                    \
        double sum = 0.0;                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = 0; j < i; j++) {                                           \
            sum += KFN((xi - data[j]) * inv_h);                                 \
        }                                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = i + 1; j < n; j++) {                                       \
            sum += KFN((xi - data[j]) * inv_h);                                 \
        }                                                                       \
                                                                                \
        out[i] = sum * inv_nh;                                                  \
    }                                                                           \
}

DEFINE_SELF_KDE_GENERIC(epanechnikov, kernel_epanechnikov)
DEFINE_SELF_KDE_GENERIC(triangular, kernel_triangular)
DEFINE_SELF_KDE_GENERIC(uniform, kernel_uniform)
DEFINE_SELF_KDE_GENERIC(cosine, kernel_cosine)

static void self_kde_generic_indirect(const double *restrict data,
                                      double *restrict out,
                                      int n, double h,
                                      kernel1d_fn kfn) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))
    for (int i = 0; i < n; i++) {
        double xi = data[i];
        double sum = 0.0;

        for (int j = 0; j < i; j++) {
            sum += kfn((xi - data[j]) * inv_h);
        }

        for (int j = i + 1; j < n; j++) {
            sum += kfn((xi - data[j]) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

void self_kde_generic(const double *restrict data, double *restrict out,
                      int n, double h, const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            self_kde_gaussian(data, out, n, h);
            return;
        case KERNEL_ID_EPANECHNIKOV:
            self_kde_generic_epanechnikov(data, out, n, h);
            return;
        case KERNEL_ID_TRIANGULAR:
            self_kde_generic_triangular(data, out, n, h);
            return;
        case KERNEL_ID_UNIFORM:
            self_kde_generic_uniform(data, out, n, h);
            return;
        case KERNEL_ID_COSINE:
            self_kde_generic_cosine(data, out, n, h);
            return;
    }

    self_kde_generic_indirect(data, out, n, h, kernel->fn);
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

#define DEFINE_EXT_KDE_GENERIC(NAME, KFN)                                       \
static void ext_kde_generic_##NAME(const double *restrict data, int n,          \
                                   const double *restrict xs,                   \
                                   double *restrict out,                        \
                                   int m, double h) {                           \
    double inv_nh = 1.0 / (n * h);                                              \
    double inv_h  = 1.0 / h;                                                    \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(static))                               \
    for (int i = 0; i < m; i++) {                                               \
        double xi  = xs[i];                                                     \
        double sum = 0.0;                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = 0; j < n; j++) {                                           \
            sum += KFN((xi - data[j]) * inv_h);                                 \
        }                                                                       \
                                                                                \
        out[i] = sum * inv_nh;                                                  \
    }                                                                           \
}

DEFINE_EXT_KDE_GENERIC(epanechnikov, kernel_epanechnikov)
DEFINE_EXT_KDE_GENERIC(triangular, kernel_triangular)
DEFINE_EXT_KDE_GENERIC(uniform, kernel_uniform)
DEFINE_EXT_KDE_GENERIC(cosine, kernel_cosine)

static void ext_kde_generic_indirect(const double *restrict data, int n,
                                     const double *restrict xs,
                                     double *restrict out,
                                     int m, double h,
                                     kernel1d_fn kfn) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            sum += kfn((xi - data[j]) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

void ext_kde_generic(const double *restrict data, int n,
                     const double *restrict xs, double *restrict out,
                     int m, double h, const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            ext_kde_gaussian(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_EPANECHNIKOV:
            ext_kde_generic_epanechnikov(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_TRIANGULAR:
            ext_kde_generic_triangular(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_UNIFORM:
            ext_kde_generic_uniform(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_COSINE:
            ext_kde_generic_cosine(data, n, xs, out, m, h);
            return;
    }

    ext_kde_generic_indirect(data, n, xs, out, m, h, kernel->fn);
}

#define DEFINE_SELF_KDE_REFLECTED(NAME, KFN)                                    \
static void self_kde_reflected_##NAME(const double *restrict data,              \
                                      double *restrict out,                     \
                                      int n, double h) {                        \
    double inv_nh = 1.0 / ((n - 1) * h);                                        \
    double inv_h  = 1.0 / h;                                                    \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))                          \
    for (int i = 0; i < n; i++) {                                               \
        double xi  = data[i];                                                   \
        double sum = 0.0;                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = 0; j < i; j++) {                                           \
            double xj = data[j];                                                \
            sum += KFN((xi - xj) * inv_h);                                      \
            sum += KFN((xi + xj) * inv_h);                                      \
            sum += KFN((xi - (2.0 - xj)) * inv_h);                              \
        }                                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = i + 1; j < n; j++) {                                       \
            double xj = data[j];                                                \
            sum += KFN((xi - xj) * inv_h);                                      \
            sum += KFN((xi + xj) * inv_h);                                      \
            sum += KFN((xi - (2.0 - xj)) * inv_h);                              \
        }                                                                       \
                                                                                \
        out[i] = sum * inv_nh;                                                  \
    }                                                                           \
}

DEFINE_SELF_KDE_REFLECTED(gaussian, kernel_gaussian)
DEFINE_SELF_KDE_REFLECTED(epanechnikov, kernel_epanechnikov)
DEFINE_SELF_KDE_REFLECTED(triangular, kernel_triangular)
DEFINE_SELF_KDE_REFLECTED(uniform, kernel_uniform)
DEFINE_SELF_KDE_REFLECTED(cosine, kernel_cosine)

static void self_kde_reflected_indirect(const double *restrict data,
                                        double *restrict out,
                                        int n, double h,
                                        kernel1d_fn kfn) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(dynamic, 64))
    for (int i = 0; i < n; i++) {
        double xi  = data[i];
        double sum = 0.0;

        for (int j = 0; j < i; j++) {
            double xj = data[j];
            sum += kfn((xi - xj) * inv_h);
            sum += kfn((xi + xj) * inv_h);
            sum += kfn((xi - (2.0 - xj)) * inv_h);
        }

        for (int j = i + 1; j < n; j++) {
            double xj = data[j];
            sum += kfn((xi - xj) * inv_h);
            sum += kfn((xi + xj) * inv_h);
            sum += kfn((xi - (2.0 - xj)) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

void self_kde_reflected(const double *restrict data, double *restrict out,
                        int n, double h, const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            self_kde_reflected_gaussian(data, out, n, h);
            return;
        case KERNEL_ID_EPANECHNIKOV:
            self_kde_reflected_epanechnikov(data, out, n, h);
            return;
        case KERNEL_ID_TRIANGULAR:
            self_kde_reflected_triangular(data, out, n, h);
            return;
        case KERNEL_ID_UNIFORM:
            self_kde_reflected_uniform(data, out, n, h);
            return;
        case KERNEL_ID_COSINE:
            self_kde_reflected_cosine(data, out, n, h);
            return;
    }

    self_kde_reflected_indirect(data, out, n, h, kernel->fn);
}

#define DEFINE_EXT_KDE_REFLECTED(NAME, KFN)                                     \
static void ext_kde_reflected_##NAME(const double *restrict data, int n,        \
                                     const double *restrict xs,                 \
                                     double *restrict out,                      \
                                     int m, double h) {                         \
    double inv_nh = 1.0 / (n * h);                                              \
    double inv_h  = 1.0 / h;                                                    \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(static))                               \
    for (int i = 0; i < m; i++) {                                               \
        double xi  = xs[i];                                                     \
        double sum = 0.0;                                                       \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                   \
        for (int j = 0; j < n; j++) {                                           \
            double xj = data[j];                                                \
            sum += KFN((xi - xj) * inv_h);                                      \
            sum += KFN((xi + xj) * inv_h);                                      \
            sum += KFN((xi - (2.0 - xj)) * inv_h);                              \
        }                                                                       \
                                                                                \
        out[i] = sum * inv_nh;                                                  \
    }                                                                           \
}

DEFINE_EXT_KDE_REFLECTED(gaussian, kernel_gaussian)
DEFINE_EXT_KDE_REFLECTED(epanechnikov, kernel_epanechnikov)
DEFINE_EXT_KDE_REFLECTED(triangular, kernel_triangular)
DEFINE_EXT_KDE_REFLECTED(uniform, kernel_uniform)
DEFINE_EXT_KDE_REFLECTED(cosine, kernel_cosine)

static void ext_kde_reflected_indirect(const double *restrict data, int n,
                                       const double *restrict xs,
                                       double *restrict out,
                                       int m, double h,
                                       kernel1d_fn kfn) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            double xj = data[j];
            sum += kfn((xi - xj) * inv_h);
            sum += kfn((xi + xj) * inv_h);
            sum += kfn((xi - (2.0 - xj)) * inv_h);
        }

        out[i] = sum * inv_nh;
    }
}

void ext_kde_reflected(const double *restrict data, int n,
                       const double *restrict xs, double *restrict out,
                       int m, double h, const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            ext_kde_reflected_gaussian(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_EPANECHNIKOV:
            ext_kde_reflected_epanechnikov(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_TRIANGULAR:
            ext_kde_reflected_triangular(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_UNIFORM:
            ext_kde_reflected_uniform(data, n, xs, out, m, h);
            return;
        case KERNEL_ID_COSINE:
            ext_kde_reflected_cosine(data, n, xs, out, m, h);
            return;
    }

    ext_kde_reflected_indirect(data, n, xs, out, m, h, kernel->fn);
}
