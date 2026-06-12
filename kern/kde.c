#include "kde.h"

#include <stdint.h>
#include <stdlib.h>

#if defined(_OPENMP)
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

static int kde_max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int kde_thread_num(void) {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static double *kde_alloc_partials(int n, int *nthreads_out) {
    int nthreads = kde_max_threads();
    *nthreads_out = nthreads;

    if (n <= 0 || nthreads <= 0) {
        return NULL;
    }

    size_t rows = (size_t)nthreads;
    size_t cols = (size_t)n;

    if (cols != 0 && rows > SIZE_MAX / cols) {
        return NULL;
    }

    size_t count = rows * cols;
    if (count > SIZE_MAX / sizeof(double)) {
        return NULL;
    }

    return (double *)calloc(count, sizeof(double));
}

#define DEFINE_SELF_KDE_SYMMETRIC(FUNC, KFN, QUAL)                             \
QUAL void FUNC(const double *restrict data, double *restrict out,              \
               int n, double h) {                                              \
    if (n <= 1) {                                                               \
        if (n == 1) {                                                           \
            out[0] = 0.0;                                                       \
        }                                                                       \
        return;                                                                 \
    }                                                                           \
                                                                                \
    double inv_nh = 1.0 / ((n - 1) * h);                                       \
    double inv_h  = 1.0 / h;                                                    \
    int nthreads  = 1;                                                          \
    double *partials = kde_alloc_partials(n, &nthreads);                        \
                                                                           \
    if (partials == NULL) {                                                     \
        OMP_PRAGMA(omp parallel for schedule(dynamic, 64))                      \
        for (int i = 0; i < n; i++) {                                          \
            double xi = data[i];                                                \
            double sum = 0.0;                                                   \
                                                                            \
            OMP_PRAGMA(omp simd reduction(+:sum))                               \
            for (int j = 0; j < i; j++) {                                      \
                sum += KFN((xi - data[j]) * inv_h);                             \
            }                                                                   \
                                                                                \
            OMP_PRAGMA(omp simd reduction(+:sum))                               \
            for (int j = i + 1; j < n; j++) {                                   \
                sum += KFN((xi - data[j]) * inv_h);                             \
            }                                                                  \
                                                                                \
            out[i] = sum * inv_nh;                                            \
        }                                                                       \
        return;                                                                 \
    }                                                                           \
                                                                                \
    OMP_PRAGMA(omp parallel)                                                    \
    {                                                                          \
        int tid = kde_thread_num();                                             \
        double *restrict local = partials + (size_t)tid * (size_t)n;            \
                                                                                \
        OMP_PRAGMA(omp for schedule(dynamic, 64))                               \
        for (int i = 0; i < n - 1; i++) {                                       \
            double xi = data[i];                                                \
            double sum_i = 0.0;                                                 \
                                                                        \
            OMP_PRAGMA(omp simd reduction(+:sum_i))                             \
            for (int j = i + 1; j < n; j++) {                                   \
                double v = KFN((xi - data[j]) * inv_h);                        \
                sum_i += v;                                                     \
                local[j] += v;                                                 \
            }                                                                  \
                                                                                \
            local[i] += sum_i;                                                 \
        }                                                                       \
    }                                                                          \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(static))                              \
    for (int i = 0; i < n; i++) {                                              \
        double sum = 0.0;                                                     \
        for (int t = 0; t < nthreads; t++) {                                    \
            sum += partials[(size_t)t * (size_t)n + (size_t)i];               \
        }                                                                       \
        out[i] = sum * inv_nh;                                                  \
    }                                                                          \
                                                                              \
    free(partials);                                                             \
}

/* self-KDE with symmetric translation-invariant kernels.
 * out[i] = (1/((n-1)*h)) * sum_{j!=i} K((xi - xj)/h)
 *
 * For symmetric K, K((xi - xj)/h) == K((xj - xi)/h), so each unordered
 * pair is evaluated once and accumulated into both output positions.
 */
DEFINE_SELF_KDE_SYMMETRIC(self_kde_gaussian, kernel_gaussian, )

void self_kde_beta(const double *restrict data, double *restrict out,
                   int n, double h) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = 0.0;
        }
        return;
    }

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

DEFINE_SELF_KDE_SYMMETRIC(self_kde_generic_epanechnikov, kernel_epanechnikov, static)
DEFINE_SELF_KDE_SYMMETRIC(self_kde_generic_triangular, kernel_triangular, static)
DEFINE_SELF_KDE_SYMMETRIC(self_kde_generic_uniform, kernel_uniform, static)
DEFINE_SELF_KDE_SYMMETRIC(self_kde_generic_cosine, kernel_cosine, static)

static void self_kde_generic_indirect(const double *restrict data,
                                      double *restrict out,
                                      int n, double h,
                                      kernel1d_fn kfn) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = 0.0;
        }
        return;
    }

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

static void self_kde_generic_symmetric_indirect(const double *restrict data,
                                                double *restrict out,
                                                int n, double h,
                                                kernel1d_fn kfn) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = 0.0;
        }
        return;
    }

    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;
    int nthreads  = 1;
    double* partials = kde_alloc_partials(n, &nthreads);

    if (partials == NULL) {
        self_kde_generic_indirect(data, out, n, h, kfn);
        return;
    }

    OMP_PRAGMA(omp parallel)
    {
        int tid = kde_thread_num();
        double* restrict local = partials + (size_t)tid * (size_t)n;

        OMP_PRAGMA(omp for schedule(dynamic, 64))
        for (int i = 0; i < n - 1; i++) {
            double xi = data[i];
            double sum_i = 0.0;

            for (int j = i + 1; j < n; j++) {
                double v = kfn((xi - data[j]) * inv_h);
                sum_i += v;
                local[j] += v;
            }

            local[i] += sum_i;
        }
    }

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; t++) {
            sum += partials[(size_t)t * (size_t)n + (size_t)i];
        }
        out[i] = sum * inv_nh;
    }

    free(partials);
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

    if (kernel->is_symmetric) {
        self_kde_generic_symmetric_indirect(data, out, n, h, kernel->fn);
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

        for (int j = 0; j < n; j++) {
            sum += kernel_beta(xi, data[j], h);
        }

        out[i] = sum * inv_n;
    }
}

#define DEFINE_EXT_KDE_GENERIC(NAME, KFN)                                   \
static void ext_kde_generic_##NAME(const double *restrict data, int n,         \
                                   const double *restrict xs,                   \
                                   double *restrict out,                       \
                                   int m, double h) {                           \
    double inv_nh = 1.0 / (n * h);                                             \
    double inv_h  = 1.0 / h;                                                    \
                                                                           \
    OMP_PRAGMA(omp parallel for schedule(static))                               \
    for (int i = 0; i < m; i++) {                                               \
        double xi  = xs[i];                                                     \
        double sum = 0.0;                                                     \
                                                                                \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = 0; j < n; j++) {                                           \
            sum += KFN((xi - data[j]) * inv_h);                              \
        }                                                                       \
                                                                     \
        out[i] = sum * inv_nh;                                                  \
    }                                                                          \
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

#define DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(FUNC, KFN, QUAL)                   \
QUAL void FUNC(const double *restrict data, double *restrict out,              \
               int n, double h) {                                              \
    if (n <= 1) {                                                               \
        if (n == 1) {                                                          \
            out[0] = 0.0;                                                       \
        }                                                                      \
        return;                                                                 \
    }                                                                           \
                                                                                \
    double inv_nh = 1.0 / ((n - 1) * h);                                        \
    double inv_h  = 1.0 / h;                                                  \
    int nthreads  = 1;                                                          \
    double *partials = kde_alloc_partials(n, &nthreads);                        \
                                                                                \
    if (partials == NULL) {                                                    \
        OMP_PRAGMA(omp parallel for schedule(dynamic, 64))                      \
        for (int i = 0; i < n; i++) {                                           \
            double xi  = data[i];                                               \
            double sum = 0.0;                                                   \
                                                                            \
            OMP_PRAGMA(omp simd reduction(+:sum))                               \
            for (int j = 0; j < i; j++) {                                      \
                double xj = data[j];                                           \
                sum += KFN((xi - xj) * inv_h);                                  \
                sum += KFN((xi + xj) * inv_h);                                 \
                sum += KFN((xi - (2.0 - xj)) * inv_h);                         \
            }                                                                   \
                                                                                \
            OMP_PRAGMA(omp simd reduction(+:sum))                              \
            for (int j = i + 1; j < n; j++) {                                   \
                double xj = data[j];                                            \
                sum += KFN((xi - xj) * inv_h);                                  \
                sum += KFN((xi + xj) * inv_h);                                  \
                sum += KFN((xi - (2.0 - xj)) * inv_h);                          \
            }                                                                   \
                                                                            \
            out[i] = sum * inv_nh;                                              \
        }                                                                       \
        return;                                                                 \
    }                                                                           \
                                                                                \
    OMP_PRAGMA(omp parallel)                                                    \
    {                                                                           \
        int tid = kde_thread_num();                                          \
        double *restrict local = partials + (size_t)tid * (size_t)n;            \
                                                                                \
        OMP_PRAGMA(omp for schedule(dynamic, 64))                               \
        for (int i = 0; i < n - 1; i++) {                                       \
            double xi = data[i];                                               \
            double sum_i = 0.0;                                                 \
                                                                                \
            OMP_PRAGMA(omp simd reduction(+:sum_i))                             \
            for (int j = i + 1; j < n; j++) {                                   \
                double xj = data[j];                                            \
                double v = KFN((xi - xj) * inv_h)                               \
                         + KFN((xi + xj) * inv_h)                              \
                         + KFN((xi + xj - 2.0) * inv_h);                        \
                sum_i += v;                                                     \
                local[j] += v;                                                  \
            }                                                                  \
                                                                                \
            local[i] += sum_i;                                                  \
        }                                                                       \
    }                                                                         \
                                                                                \
    OMP_PRAGMA(omp parallel for schedule(static))                               \
    for (int i = 0; i < n; i++) {                                               \
        double sum = 0.0;                                                       \
        for (int t = 0; t < nthreads; t++) {                                    \
            sum += partials[(size_t)t * (size_t)n + (size_t)i];                \
        }                                                                       \
        out[i] = sum * inv_nh;                                                  \
    }                                                                         \
                                                                                \
    free(partials);                                                             \
}

DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(self_kde_reflected_gaussian, kernel_gaussian, static)
DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(self_kde_reflected_epanechnikov, kernel_epanechnikov, static)
DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(self_kde_reflected_triangular, kernel_triangular, static)
DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(self_kde_reflected_uniform, kernel_uniform, static)
DEFINE_SELF_KDE_REFLECTED_SYMMETRIC(self_kde_reflected_cosine, kernel_cosine, static)

static void self_kde_reflected_indirect(const double *restrict data,
                                        double *restrict out,
                                        int n, double h,
                                        kernel1d_fn kfn) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = 0.0;
        }
        return;
    }

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

static void self_kde_reflected_symmetric_indirect(const double *restrict data,
                                                  double *restrict out,
                                                  int n, double h,
                                                  kernel1d_fn kfn) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = 0.0;
        }
        return;
    }

    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;
    int nthreads  = 1;
    double *partials = kde_alloc_partials(n, &nthreads);

    if (partials == NULL) {
        self_kde_reflected_indirect(data, out, n, h, kfn);
        return;
    }

    OMP_PRAGMA(omp parallel)
    {
        int tid = kde_thread_num();
        double *restrict local = partials + (size_t)tid * (size_t)n;

        OMP_PRAGMA(omp for schedule(dynamic, 64))
        for (int i = 0; i < n - 1; i++) {
            double xi = data[i];
            double sum_i = 0.0;

            for (int j = i + 1; j < n; j++) {
                double xj = data[j];
                double v = kfn((xi - xj) * inv_h)
                         + kfn((xi + xj) * inv_h)
                         + kfn((xi + xj - 2.0) * inv_h);
                sum_i += v;
                local[j] += v;
            }

            local[i] += sum_i;
        }
    }

    OMP_PRAGMA(omp parallel for schedule(static))
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; t++) {
            sum += partials[(size_t)t * (size_t)n + (size_t)i];
        }
        out[i] = sum * inv_nh;
    }

    free(partials);
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

    if (kernel->is_symmetric) {
        self_kde_reflected_symmetric_indirect(data, out, n, h, kernel->fn);
        return;
    }

    self_kde_reflected_indirect(data, out, n, h, kernel->fn);
}

#define DEFINE_EXT_KDE_REFLECTED(NAME, KFN)                                    \
static void ext_kde_reflected_##NAME(const double *restrict data, int n,        \
                                     const double *restrict xs,                 \
                                     double *restrict out,                      \
                                     int m, double h) {                        \
    double inv_nh = 1.0 / (n * h);                                              \
    double inv_h  = 1.0 / h;                                                    \
                                                                            \
    OMP_PRAGMA(omp parallel for schedule(static))                               \
    for (int i = 0; i < m; i++) {                                               \
        double xi  = xs[i];                                                  \
        double sum = 0.0;                                                       \
                                                                               \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = 0; j < n; j++) {                                          \
            double xj = data[j];                                              \
            sum += KFN((xi - xj) * inv_h);                                      \
            sum += KFN((xi + xj) * inv_h);                                    \
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
