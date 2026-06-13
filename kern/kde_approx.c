#include "kde_approx.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_OPENMP)
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

#define APPROX_HIGH_MEMORY_LIMIT ((size_t)256 * 1024 * 1024)

static int max_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int thread_num(void) {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static int lower_bound_double(const double *data, int n, double value) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (data[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int upper_bound_double(const double *data, int n, double value) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (data[mid] <= value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline double fast_exp_schraudolph(double x) {
    union {
        uint64_t bits;
        double value;
    } result;
    if (x < -700.0) return 0.0;
    result.bits = ((uint64_t)(1512775.0 * x + 1072632447.0)) << 32;
    return result.value;
}

static inline double kernel_gaussian_fast(double u) {
    return MATHCONST_1_SQRT2PI * fast_exp_schraudolph(-0.5 * u * u);
}

static void make_self_bounds(const double *data, int n, double distance,
                             int max_neighbors, int *left, int *right) {
    int left_cursor = 0;
    int right_cursor = 0;

    for (int i = 0; i < n; i++) {
        while (left_cursor < i && data[i] - data[left_cursor] > distance) {
            left_cursor++;
        }
        if (right_cursor < i + 1) right_cursor = i + 1;
        while (right_cursor < n && data[right_cursor] - data[i] <= distance) {
            right_cursor++;
        }

        left[i] = left_cursor;
        right[i] = right_cursor;
        if (max_neighbors > 0) {
            int first = i - max_neighbors;
            int last = i + max_neighbors + 1;
            if (first > left[i]) left[i] = first;
            if (last < right[i]) right[i] = last;
        }
    }
}

static int queries_are_sorted(const double *xs, int m) {
    for (int i = 1; i < m; i++) {
        if (xs[i] < xs[i - 1]) return 0;
    }
    return 1;
}

static void make_ext_bounds(const double *data, int n,
                            const double *xs, int m, double distance,
                            int max_neighbors, int *left, int *right) {
    if (queries_are_sorted(xs, m)) {
        int left_cursor = 0;
        int right_cursor = 0;
        int center = 0;
        for (int i = 0; i < m; i++) {
            while (left_cursor < n && data[left_cursor] < xs[i] - distance) {
                left_cursor++;
            }
            if (right_cursor < left_cursor) right_cursor = left_cursor;
            while (right_cursor < n && data[right_cursor] <= xs[i] + distance) {
                right_cursor++;
            }
            left[i] = left_cursor;
            right[i] = right_cursor;

            if (max_neighbors > 0) {
                while (center < n && data[center] < xs[i]) center++;
                int first = center - max_neighbors;
                int last = center + max_neighbors;
                if (first < 0) first = 0;
                if (last > n) last = n;
                if (first > left[i]) left[i] = first;
                if (last < right[i]) right[i] = last;
            }
        }
        return;
    }

    OMP_PRAGMA(omp parallel for schedule(static) if(m > 512))
    for (int i = 0; i < m; i++) {
        int first = lower_bound_double(data, n, xs[i] - distance);
        int last = upper_bound_double(data, n, xs[i] + distance);
        if (max_neighbors > 0) {
            int center = lower_bound_double(data, n, xs[i]);
            int knn_first = center - max_neighbors;
            int knn_last = center + max_neighbors;
            if (knn_first < 0) knn_first = 0;
            if (knn_last > n) knn_last = n;
            if (knn_first > first) first = knn_first;
            if (knn_last < last) last = knn_last;
        }
        left[i] = first;
        right[i] = last;
    }
}

#define DEFINE_SELF_LOW(NAME, KFN)                                             \
static void self_low_##NAME(const double *restrict data,                       \
                            double *restrict out, int n, double inv_h,          \
                            double scale, const int *left, const int *right) {  \
    OMP_PRAGMA(omp parallel for schedule(guided, 64) if(n > 512))              \
    for (int i = 0; i < n; i++) {                                              \
        double sum = 0.0;                                                       \
        double x = data[i];                                                     \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = left[i]; j < i; j++) sum += KFN((x - data[j]) * inv_h);   \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = i + 1; j < right[i]; j++)                                 \
            sum += KFN((x - data[j]) * inv_h);                                 \
        out[i] = sum * scale;                                                   \
    }                                                                           \
}

#define DEFINE_SELF_HIGH(NAME, KFN)                                            \
static int self_high_##NAME(const double *restrict data,                       \
                            double *restrict out, int n, double inv_h,          \
                            double scale, const int *right) {                   \
    int threads = max_threads();                                                \
    size_t count = (size_t)threads * n;                                         \
    if (n != 0 && count / (size_t)n != (size_t)threads) return 0;              \
    if (count > SIZE_MAX / sizeof(double)) return 0;                            \
    double *partials = (double *)calloc(count, sizeof(double));                 \
    if (!partials) return 0;                                                    \
    OMP_PRAGMA(omp parallel if(n > 256))                                        \
    {                                                                           \
        double *local = partials + (size_t)thread_num() * n;                    \
        OMP_PRAGMA(omp for schedule(guided, 64))                               \
        for (int i = 0; i < n - 1; i++) {                                      \
            double sum = 0.0;                                                   \
            double x = data[i];                                                 \
            OMP_PRAGMA(omp simd reduction(+:sum))                              \
            for (int j = i + 1; j < right[i]; j++) {                           \
                double value = KFN((x - data[j]) * inv_h);                     \
                sum += value;                                                   \
                local[j] += value;                                              \
            }                                                                   \
            local[i] += sum;                                                    \
        }                                                                       \
    }                                                                           \
    OMP_PRAGMA(omp parallel for schedule(static) if(n > 512))                  \
    for (int i = 0; i < n; i++) {                                              \
        double sum = 0.0;                                                       \
        for (int t = 0; t < threads; t++) sum += partials[(size_t)t * n + i];  \
        out[i] = sum * scale;                                                   \
    }                                                                           \
    free(partials);                                                             \
    return 1;                                                                   \
}

#define DEFINE_EXT_BOUNDS(NAME, KFN)                                           \
static void ext_bounds_##NAME(const double *restrict data,                     \
                              const double *restrict xs, double *restrict out, \
                              int m, double inv_h, double scale,                \
                              const int *left, const int *right) {              \
    OMP_PRAGMA(omp parallel for schedule(guided, 64) if(m > 256))              \
    for (int i = 0; i < m; i++) {                                              \
        double sum = 0.0;                                                       \
        double x = xs[i];                                                       \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = left[i]; j < right[i]; j++)                               \
            sum += KFN((x - data[j]) * inv_h);                                 \
        out[i] = sum * scale;                                                   \
    }                                                                           \
}

#define DEFINE_EXT_LOW(NAME, KFN)                                              \
static void ext_low_##NAME(const double *restrict data, int n,                 \
                           const double *restrict xs, double *restrict out,    \
                           int m, double distance, int max_neighbors,           \
                           double inv_h, double scale) {                        \
    OMP_PRAGMA(omp parallel for schedule(dynamic, 64) if(m > 256))              \
    for (int i = 0; i < m; i++) {                                              \
        double x = xs[i];                                                       \
        int first = lower_bound_double(data, n, x - distance);                 \
        int last = upper_bound_double(data, n, x + distance);                  \
        if (max_neighbors > 0) {                                                \
            int center = lower_bound_double(data, n, x);                       \
            int knn_first = center - max_neighbors;                            \
            int knn_last = center + max_neighbors;                             \
            if (knn_first < 0) knn_first = 0;                                  \
            if (knn_last > n) knn_last = n;                                    \
            if (knn_first > first) first = knn_first;                          \
            if (knn_last < last) last = knn_last;                              \
        }                                                                       \
        double sum = 0.0;                                                       \
        OMP_PRAGMA(omp simd reduction(+:sum))                                  \
        for (int j = first; j < last; j++) sum += KFN((x - data[j]) * inv_h); \
        out[i] = sum * scale;                                                   \
    }                                                                           \
}

#define DEFINE_APPROX_KERNEL(NAME, KFN)                                        \
    DEFINE_SELF_LOW(NAME, KFN)                                                 \
    DEFINE_SELF_HIGH(NAME, KFN)                                                \
    DEFINE_EXT_BOUNDS(NAME, KFN)                                               \
    DEFINE_EXT_LOW(NAME, KFN)

DEFINE_APPROX_KERNEL(gaussian, kernel_gaussian)
DEFINE_APPROX_KERNEL(gaussian_fast, kernel_gaussian_fast)
DEFINE_APPROX_KERNEL(epanechnikov, kernel_epanechnikov)
DEFINE_APPROX_KERNEL(triangular, kernel_triangular)
DEFINE_APPROX_KERNEL(uniform, kernel_uniform)
DEFINE_APPROX_KERNEL(cosine, kernel_cosine)

#define DISPATCH_SELF(NAME)                                                     \
    do {                                                                        \
        if (use_high && self_high_##NAME(sorted_data, out, n, inv_h, scale,    \
                                          right)) break;                        \
        self_low_##NAME(sorted_data, out, n, inv_h, scale, left, right);       \
    } while (0)

void approx_self_kde(const double *restrict sorted_data,
                     double *restrict out, int n, double h,
                     const kernel1d_info *kernel, double cutoff,
                     int max_neighbors, int fast_gaussian, int memory_mode) {
    if (n <= 1) {
        if (n == 1) out[0] = 0.0;
        return;
    }

    int *bounds = (int *)malloc((size_t)2 * n * sizeof(int));
    if (!bounds) return;
    int *left = bounds;
    int *right = bounds + n;
    double used_cutoff = cutoff > 0.0 ? cutoff : kernel->default_cutoff;
    make_self_bounds(sorted_data, n, used_cutoff * h,
                     max_neighbors, left, right);

    int use_high = memory_mode == KDE_APPROX_MEMORY_HIGH;
    if (memory_mode == KDE_APPROX_MEMORY_AUTO) {
        size_t bytes = (size_t)max_threads() * n * sizeof(double);
        long long pairs = 0;
        for (int i = 0; i < n; i++) pairs += right[i] - i - 1;
        use_high = bytes <= APPROX_HIGH_MEMORY_LIMIT && pairs > (long long)n * 96;
    }

    double inv_h = 1.0 / h;
    double scale = 1.0 / ((n - 1) * h);
    if (fast_gaussian && kernel->id == KERNEL_ID_GAUSSIAN) {
        DISPATCH_SELF(gaussian_fast);
    } else {
        switch (kernel->id) {
            case KERNEL_ID_GAUSSIAN: DISPATCH_SELF(gaussian); break;
            case KERNEL_ID_EPANECHNIKOV: DISPATCH_SELF(epanechnikov); break;
            case KERNEL_ID_TRIANGULAR: DISPATCH_SELF(triangular); break;
            case KERNEL_ID_UNIFORM: DISPATCH_SELF(uniform); break;
            case KERNEL_ID_COSINE: DISPATCH_SELF(cosine); break;
        }
    }
    free(bounds);
}

#define DISPATCH_EXT(NAME)                                                      \
    do {                                                                        \
        if (use_bounds) ext_bounds_##NAME(sorted_data, xs, out, m, inv_h,      \
                                           scale, left, right);                 \
        else ext_low_##NAME(sorted_data, n, xs, out, m, distance,              \
                            max_neighbors, inv_h, scale);                       \
    } while (0)

void approx_ext_kde(const double *restrict sorted_data, int n,
                    const double *restrict xs, double *restrict out, int m,
                    double h, const kernel1d_info *kernel, double cutoff,
                    int max_neighbors, int fast_gaussian, int memory_mode) {
    double used_cutoff = cutoff > 0.0 ? cutoff : kernel->default_cutoff;
    double distance = used_cutoff * h;
    double inv_h = 1.0 / h;
    double scale = 1.0 / (n * h);
    int use_bounds = memory_mode == KDE_APPROX_MEMORY_HIGH
                  || (memory_mode == KDE_APPROX_MEMORY_AUTO && m >= 256
                      && queries_are_sorted(xs, m));
    int *bounds = NULL;
    int *left = NULL;
    int *right = NULL;

    if (use_bounds) {
        bounds = (int *)malloc((size_t)2 * m * sizeof(int));
        if (bounds) {
            left = bounds;
            right = bounds + m;
            make_ext_bounds(sorted_data, n, xs, m, distance,
                            max_neighbors, left, right);
        } else {
            use_bounds = 0;
        }
    }

    if (fast_gaussian && kernel->id == KERNEL_ID_GAUSSIAN) {
        DISPATCH_EXT(gaussian_fast);
    } else {
        switch (kernel->id) {
            case KERNEL_ID_GAUSSIAN: DISPATCH_EXT(gaussian); break;
            case KERNEL_ID_EPANECHNIKOV: DISPATCH_EXT(epanechnikov); break;
            case KERNEL_ID_TRIANGULAR: DISPATCH_EXT(triangular); break;
            case KERNEL_ID_UNIFORM: DISPATCH_EXT(uniform); break;
            case KERNEL_ID_COSINE: DISPATCH_EXT(cosine); break;
        }
    }
    free(bounds);
}
