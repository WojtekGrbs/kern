#include "bandwidth.h"
#include "kde.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

static int bandwidth_max_threads(void) {
#if defined(_OPEopNMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Define compact kernels through DEFINE_SELF_SERIAL_COMPACT and the others
// through DEFINE_SELF_SERIAL.

//compact kernel is K(u)=0 iff |u|>1

// I should also probably move it outside/let kde.c allow 
// parameter on serial/multithreaded

#define DEFINE_SELF_SERIAL(NAME, KFN)                                          \
static void self_serial_##NAME(const double *restrict data,                    \
                               double *restrict out, int n, double h) {         \
    memset(out, 0, (size_t)n * sizeof(double));                                \
    double inv_h = 1.0 / h;                                                    \
    double scale = 1.0 / ((n - 1) * h);                                        \
    for (int i = 0; i < n - 1; i++) {                                          \
        double xi = data[i];                                                    \
        double sum = 0.0;                                                       \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = i + 1; j < n; j++) {                                      \
            double value = KFN((xi - data[j]) * inv_h);                        \
            sum += value;                                                       \
            out[j] += value;                                                    \
        }                                                                       \
        out[i] += sum;                                                          \
    }                                                                           \
    OMP_PRAGMA(omp simd)                                                       \
    for (int i = 0; i < n; i++) out[i] *= scale;                               \
}

// TODO: implement compact sorted traversal in the regular KDE.
// assumes a sorted array from left to right.

#define DEFINE_SELF_SERIAL_COMPACT(NAME, KFN)                                  \
static void self_serial_##NAME(const double *restrict data,                    \
                               double *restrict out, int n, double h) {         \
    memset(out, 0, (size_t)n * sizeof(double));                                \
    double inv_h = 1.0 / h;                                                    \
    double scale = 1.0 / ((n - 1) * h);                                        \
    int right = 1;                                                              \
    for (int i = 0; i < n - 1; i++) {                                          \
        if (right < i + 1) right = i + 1;                                      \
        while (right < n && data[right] - data[i] <= h) right++;               \
        double xi = data[i];                                                    \
        double sum = 0.0;                                                       \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = i + 1; j < right; j++) {                                  \
            double value = KFN((xi - data[j]) * inv_h);                        \
            sum += value;                                                       \
            out[j] += value;                                                    \
        }                                                                       \
        out[i] += sum;                                                          \
    }                                                                           \
    OMP_PRAGMA(omp simd)                                                       \
    for (int i = 0; i < n; i++) out[i] *= scale;                               \
}

#define DEFINE_EXT_SERIAL(NAME, KFN)                                           \
static void ext_serial_##NAME(const double *restrict data, int n,              \
                              const double *restrict xs,                       \
                              double *restrict out, int m, double h) {          \
    double inv_h = 1.0 / h;                                                    \
    double scale = 1.0 / (n * h);                                              \
    for (int i = 0; i < m; i++) {                                              \
        double sum = 0.0;                                                       \
        double x = xs[i];                                                       \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = 0; j < n; j++) {                                          \
            sum += KFN((x - data[j]) * inv_h);                                 \
        }                                                                       \
        out[i] = sum * scale;                                                   \
    }                                                                           \
}

DEFINE_SELF_SERIAL(gaussian, kernel_gaussian)
DEFINE_SELF_SERIAL_COMPACT(epanechnikov, kernel_epanechnikov)
DEFINE_SELF_SERIAL_COMPACT(triangular, kernel_triangular)
DEFINE_SELF_SERIAL_COMPACT(uniform, kernel_uniform)
DEFINE_SELF_SERIAL_COMPACT(cosine, kernel_cosine)

DEFINE_EXT_SERIAL(gaussian, kernel_gaussian)
DEFINE_EXT_SERIAL(epanechnikov, kernel_epanechnikov)
DEFINE_EXT_SERIAL(triangular, kernel_triangular)
DEFINE_EXT_SERIAL(uniform, kernel_uniform)
DEFINE_EXT_SERIAL(cosine, kernel_cosine)

static void self_serial(const double *data, double *out, int n, double h,
                        const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            self_serial_gaussian(data, out, n, h); return;
        case KERNEL_ID_EPANECHNIKOV:
            self_serial_epanechnikov(data, out, n, h); return;
        case KERNEL_ID_TRIANGULAR:
            self_serial_triangular(data, out, n, h); return;
        case KERNEL_ID_UNIFORM:
            self_serial_uniform(data, out, n, h); return;
        case KERNEL_ID_COSINE:
            self_serial_cosine(data, out, n, h); return;
    }
}

static void ext_serial(const double *data, int n, const double *xs,
                       double *out, int m, double h,
                       const kernel1d_info *kernel) {
    switch (kernel->id) {
        case KERNEL_ID_GAUSSIAN:
            ext_serial_gaussian(data, n, xs, out, m, h); return;
        case KERNEL_ID_EPANECHNIKOV:
            ext_serial_epanechnikov(data, n, xs, out, m, h); return;
        case KERNEL_ID_TRIANGULAR:
            ext_serial_triangular(data, n, xs, out, m, h); return;
        case KERNEL_ID_UNIFORM:
            ext_serial_uniform(data, n, xs, out, m, h); return;
        case KERNEL_ID_COSINE:
            ext_serial_cosine(data, n, xs, out, m, h); return;
    }
}

static double average_log_density(const double *density, int n) {
    double score = 0.0;
    OMP_PRAGMA(omp simd reduction(+:score))
    for (int i = 0; i < n; i++) {
        score += density[i] > 1e-300 ? log(density[i]) : -700.0;
    }
    return score / n;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double loo_score_serial(const double *data, int n, double h,
                               const kernel1d_info *kernel,
                               double *workspace) {
    self_serial(data, workspace, n, h, kernel);
    return average_log_density(workspace, n);
}

static double kfold_score_serial(const double *data, int n, int k_folds,
                                 double h, const kernel1d_info *kernel,
                                 double *workspace) {
    double *train = workspace;
    double *test_density = train + n;
    double total_score = 0.0;

    for (int fold = 0; fold < k_folds; fold++) {
        int test_start = (fold * n) / k_folds;
        int test_end = ((fold + 1) * n) / k_folds;
        int n_test = test_end - test_start;
        int n_train = n - n_test;
        int train_index = 0;

        for (int i = 0; i < test_start; i++) train[train_index++] = data[i];
        for (int i = test_end; i < n; i++) train[train_index++] = data[i];

        ext_serial(train, n_train, data + test_start, test_density,
                   n_test, h, kernel);
        total_score += average_log_density(test_density, n_test) * n_test;
    }
    return total_score / n;
}

static double score_serial(const double *data, int n, double h, int k_folds,
                           const kernel1d_info *kernel, double *workspace) {
    if (k_folds > 1) {
        return kfold_score_serial(data, n, k_folds, h, kernel, workspace);
    }
    return loo_score_serial(data, n, h, kernel, workspace);
}

double loo_score_kernel(const double *restrict data, int n, double h,
                        const kernel1d_info *kernel) {
    if (n <= 1 || h <= 0.0 || !kernel) return -DBL_MAX;

    double *workspace = (double *)malloc((size_t)n * sizeof(double));
    if (!workspace) return -DBL_MAX;

    self_kde_generic(data, workspace, n, h, kernel);
    double score = average_log_density(workspace, n);

    free(workspace);
    return score;
}

double kfold_score_kernel(const double *restrict data, int n,
                          int k_folds, double h,
                          const kernel1d_info *kernel) {
    if (n <= 1 || k_folds < 2 || k_folds > n || h <= 0.0 || !kernel) {
        return -DBL_MAX;
    }

    double *workspace = (double *)malloc((size_t)2 * n * sizeof(double));
    if (!workspace) return -DBL_MAX;

    double *train = workspace;
    double *test_density = train + n;
    double score = 0.0;

    for (int fold = 0; fold < k_folds; fold++) {
        int test_start = (fold * n) / k_folds;
        int test_end = ((fold + 1) * n) / k_folds;
        int n_test = test_end - test_start;
        int n_train = n - n_test;
        int train_index = 0;

        for (int i = 0; i < test_start; i++) train[train_index++] = data[i];
        for (int i = test_end; i < n; i++) train[train_index++] = data[i];

        ext_kde_generic(train, n_train, data + test_start, test_density,
                        n_test, h, kernel);
        score += average_log_density(test_density, n_test) * n_test;
    }

    free(workspace);
    return score / n;
}

void bandwidth_score_grid(const double *restrict data, int n,
                          const double *restrict h_grid, int n_bandwidths,
                          const kernel1d_info *kernel, int k_folds,
                          int parallel_mode,
                          double *restrict scores) {
    int chosen_parallel = parallel_mode;
    if (chosen_parallel == BANDWIDTH_PARALLEL_AUTO) {
        int threads = bandwidth_max_threads();
        // a short grid cannot keep all threads busy on the outside
        chosen_parallel = n_bandwidths == 1
                        ? BANDWIDTH_PARALLEL_EVALUATION
                        : n_bandwidths >= threads || n < 4096
                            ? BANDWIDTH_PARALLEL_GRID
                            : BANDWIDTH_PARALLEL_EVALUATION;
    }

    // if parallel over evaluation, just call the kde.c functions
    if (chosen_parallel == BANDWIDTH_PARALLEL_EVALUATION) {
        for (int hi = 0; hi < n_bandwidths; hi++) {
            scores[hi] = k_folds > 1
                       ? kfold_score_kernel(data, n, k_folds, h_grid[hi],
                                            kernel)
                       : loo_score_kernel(data, n, h_grid[hi], kernel);
        }
        return;
    }

    // if parallel over h, call the serial evaluation (Defined here - move to kde.c?)
    
    // for loo, sort the data first (for compact)
    const double *used_data = data;
    double *sorted_data = NULL;
    if (k_folds == 1 && kernel->id != KERNEL_ID_GAUSSIAN) {
        sorted_data = (double *)malloc((size_t)n * sizeof(double));
        if (!sorted_data) {
            for (int hi = 0; hi < n_bandwidths; hi++) {
                scores[hi] = -DBL_MAX;
            }
            return;
        }
        memcpy(sorted_data, data, (size_t)n * sizeof(double));
        qsort(sorted_data, (size_t)n, sizeof(double), compare_double);
        used_data = sorted_data;
    }

    size_t workspace_count = (size_t)2 * n;
    OMP_PRAGMA(omp parallel)
    {
        double *workspace =
            (double *)malloc(workspace_count * sizeof(double));

        OMP_PRAGMA(omp for schedule(static))
        for (int hi = 0; hi < n_bandwidths; hi++) {
            scores[hi] = workspace
                       ? score_serial(used_data, n, h_grid[hi], k_folds,
                                      kernel, workspace)
                       : -DBL_MAX;
        }
        free(workspace);
    }
    free(sorted_data);
}

double loo_score(const double *restrict data, int n, double h) {
    return loo_score_kernel(data, n, h, get_kernel("gaussian"));
}

double kfold_score(const double *restrict data, int n,
                   int k_folds, double h) {
    return kfold_score_kernel(data, n, k_folds, h, get_kernel("gaussian"));
}
