#include "kde_multivariate.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_OPENMP)
  #include <omp.h>
  #define OMP_PRAGMA(x) _Pragma(#x)
#else
  #define OMP_PRAGMA(x)
#endif

#define MV_MAX_QUERY_BLOCK 64
#define MV_DATA_BLOCK 128
#define MV_PARTIALS_LIMIT ((size_t)256 * 1024 * 1024) //limits so that n_threads*n does not explode

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

static int checked_block_size(int block_size) {
    if (block_size < 1) return 16;
    if (block_size > MV_MAX_QUERY_BLOCK) return MV_MAX_QUERY_BLOCK;
    return block_size;
}

static double inverse_h_power(double h, int dimensions) {
    double value = 1.0;
    for (int d = 0; d < dimensions; d++) value /= h;
    return value;
}

static inline double gaussian_pair(const double *restrict x,
                                   const double *restrict point,
                                   int dimensions, double inv_h) {
    double squared = 0.0;
    OMP_PRAGMA(omp simd reduction(+:squared))
    for (int d = 0; d < dimensions; d++) {
        double delta = (x[d] - point[d]) * inv_h;
        squared += delta * delta;
    }
    return exp(-0.5 * squared);
}

#define DEFINE_PRODUCT_PAIR(NAME, KFN)                                         \
static inline double pair_##NAME(const double *restrict x,                     \
                                 const double *restrict point,                 \
                                 int dimensions, double inv_h) {               \
    double value = 1.0;                                                         \
    OMP_PRAGMA(omp simd reduction(*:value))                                    \
    for (int d = 0; d < dimensions; d++) {                                     \
        value *= KFN((x[d] - point[d]) * inv_h);                               \
    }                                                                           \
    return value;                                                               \
}

DEFINE_PRODUCT_PAIR(epanechnikov, kernel_epanechnikov)
DEFINE_PRODUCT_PAIR(triangular, kernel_triangular)
DEFINE_PRODUCT_PAIR(uniform, kernel_uniform)
DEFINE_PRODUCT_PAIR(cosine, kernel_cosine)

#define DEFINE_BLOCKED_EXT(NAME, PAIR_FN)                                      \
static void blocked_ext_##NAME(const double *restrict data, int n,             \
                               int dimensions, const double *restrict xs,      \
                               double *restrict out, int m, double inv_h,       \
                               double scale, int query_block) {                \
    int block_count = (m + query_block - 1) / query_block;                     \
    long long work = (long long)n * m * dimensions;                            \
    (void)work;                                                                 \
    OMP_PRAGMA(omp parallel for schedule(static))             \
    for (int block = 0; block < block_count; block++) {                        \
        int first_query = block * query_block;                                  \
        int query_count = m - first_query;                                      \
        if (query_count > query_block) query_count = query_block;              \
        double sums[MV_MAX_QUERY_BLOCK] = {0.0};                               \
        for (int first_data = 0; first_data < n; first_data += MV_DATA_BLOCK) {\
            int last_data = first_data + MV_DATA_BLOCK;                         \
            if (last_data > n) last_data = n;                                  \
            for (int q = 0; q < query_count; q++) {                            \
                const double *x = xs + (size_t)(first_query + q) * dimensions; \
                double sum = sums[q];                                           \
                for (int j = first_data; j < last_data; j++) {                 \
                    const double *point = data + (size_t)j * dimensions;        \
                    sum += PAIR_FN(x, point, dimensions, inv_h);               \
                }                                                               \
                sums[q] = sum;                                                  \
            }                                                                   \
        }                                                                       \
        OMP_PRAGMA(omp simd)                                                   \
        for (int q = 0; q < query_count; q++) {                                \
            out[first_query + q] = sums[q] * scale;                            \
        }                                                                       \
    }                                                                           \
}

#define DEFINE_BLOCKED_SELF(NAME, PAIR_FN)                                     \
static void blocked_self_##NAME(const double *restrict data,                   \
                                double *restrict out, int n, int dimensions,   \
                                double inv_h, double scale, int query_block) { \
    int block_count = (n + query_block - 1) / query_block;                     \
    long long work = (long long)n * n * dimensions;                            \
    (void)work;                                                                 \
    OMP_PRAGMA(omp parallel for schedule(static))             \
    for (int block = 0; block < block_count; block++) {                        \
        int first_query = block * query_block;                                  \
        int query_count = n - first_query;                                      \
        if (query_count > query_block) query_count = query_block;              \
        double sums[MV_MAX_QUERY_BLOCK] = {0.0};                               \
        for (int first_data = 0; first_data < n; first_data += MV_DATA_BLOCK) {\
            int last_data = first_data + MV_DATA_BLOCK;                         \
            if (last_data > n) last_data = n;                                  \
            for (int q = 0; q < query_count; q++) {                            \
                int i = first_query + q;                                        \
                const double* x = data + (size_t)i * dimensions;               \
                double sum = sums[q];                                           \
                for (int j = first_data; j < last_data; j++) {                 \
                    if (i == j) continue;                                       \
                    const double* point = data + (size_t)j * dimensions;        \
                    sum += PAIR_FN(x, point, dimensions, inv_h);               \
                }                                                               \
                sums[q] = sum;                                                  \
            }                                                                   \
        }                                                                       \
        OMP_PRAGMA(omp simd)                                                   \
        for (int q = 0; q < query_count; q++) {                                \
            out[first_query + q] = sums[q] * scale;                            \
        }                                                                       \
    }                                                                           \
}

#define DEFINE_SYMMETRIC_SELF(NAME, PAIR_FN)                                   \
static int symmetric_self_##NAME(const double *restrict data,                  \
                                 double *restrict out, int n, int dimensions,  \
                                 double inv_h, double scale, int query_block) {\
    int threads = max_threads();                                                \
    size_t count = (size_t)threads * n;                                         \
    if (n != 0 && count / (size_t)n != (size_t)threads) return 0;              \
    if (count > SIZE_MAX / sizeof(double)) return 0;                            \
    if (count > MV_PARTIALS_LIMIT / sizeof(double)) return 0;                  \
    double *partials = (double *)calloc(count, sizeof(double));                 \
    if (!partials) return 0;                                                    \
    int block_count = (n + query_block - 1) / query_block;                     \
    OMP_PRAGMA(omp parallel if(n > 256))                                        \
    {                                                                           \
        double *local = partials + (size_t)thread_num() * n;                    \
        OMP_PRAGMA(omp for schedule(dynamic, 1))                               \
        for (int block = 0; block < block_count; block++) {                    \
            int first_query = block * query_block;                              \
            int last_query = first_query + query_block;                         \
            if (last_query > n) last_query = n;                                \
            for (int first_data = first_query; first_data < n;                 \
                 first_data += MV_DATA_BLOCK) {                                 \
                int last_data = first_data + MV_DATA_BLOCK;                     \
                if (last_data > n) last_data = n;                              \
                for (int i = first_query; i < last_query; i++) {               \
                    int first_j = first_data;                                   \
                    if (first_j <= i) first_j = i + 1;                         \
                    const double *x = data + (size_t)i * dimensions;           \
                    double sum = 0.0;                                           \
                    for (int j = first_j; j < last_data; j++) {                \
                        const double *point = data + (size_t)j * dimensions;    \
                        double value = PAIR_FN(x, point, dimensions, inv_h);   \
                        sum += value;                                           \
                        local[j] += value;                                      \
                    }                                                           \
                    local[i] += sum;                                            \
                }                                                               \
            }                                                                   \
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

DEFINE_BLOCKED_EXT(gaussian, gaussian_pair)
DEFINE_BLOCKED_EXT(epanechnikov, pair_epanechnikov)
DEFINE_BLOCKED_EXT(triangular, pair_triangular)
DEFINE_BLOCKED_EXT(uniform, pair_uniform)
DEFINE_BLOCKED_EXT(cosine, pair_cosine)

DEFINE_BLOCKED_SELF(gaussian, gaussian_pair)
DEFINE_BLOCKED_SELF(epanechnikov, pair_epanechnikov)
DEFINE_BLOCKED_SELF(triangular, pair_triangular)
DEFINE_BLOCKED_SELF(uniform, pair_uniform)
DEFINE_BLOCKED_SELF(cosine, pair_cosine)

DEFINE_SYMMETRIC_SELF(gaussian, gaussian_pair)
DEFINE_SYMMETRIC_SELF(epanechnikov, pair_epanechnikov)
DEFINE_SYMMETRIC_SELF(triangular, pair_triangular)
DEFINE_SYMMETRIC_SELF(uniform, pair_uniform)
DEFINE_SYMMETRIC_SELF(cosine, pair_cosine)

// only apis below
void multivariate_kde(const double *restrict data, int n, int dimensions,
                      const double *restrict xs, double *restrict out, int m,
                      double h, const kernel1d_info *kernel, int block_size) {
    double inv_h = 1.0 / h;
    double scale = inverse_h_power(h, dimensions) / n;
    int query_block = checked_block_size(block_size);

    if (kernel->id == KERNEL_ID_GAUSSIAN) {
        double normalizer = 1.0;
        for (int d = 0; d < dimensions; d++) {
            normalizer *= MATHCONST_1_SQRT2PI;
        }
        blocked_ext_gaussian(data, n, dimensions, xs, out, m, inv_h,
                             scale * normalizer, query_block);
        return;
    }

    switch (kernel->id) {
        case KERNEL_ID_EPANECHNIKOV:
            blocked_ext_epanechnikov(data, n, dimensions, xs, out, m, inv_h,
                                     scale, query_block); return;
        case KERNEL_ID_TRIANGULAR:
            blocked_ext_triangular(data, n, dimensions, xs, out, m, inv_h,
                                   scale, query_block); return;
        case KERNEL_ID_UNIFORM:
            blocked_ext_uniform(data, n, dimensions, xs, out, m, inv_h,
                                scale, query_block); return;
        case KERNEL_ID_COSINE:
            blocked_ext_cosine(data, n, dimensions, xs, out, m, inv_h,
                               scale, query_block); return;
        case KERNEL_ID_GAUSSIAN:
            return;
    }
}

void multivariate_self_kde(const double *restrict data, double *restrict out,
                           int n, int dimensions, double h,
                           const kernel1d_info *kernel, int block_size) {
    if (n <= 1) {
        if (n == 1) out[0] = 0.0;
        return;
    }

    double inv_h = 1.0 / h;
    double scale = inverse_h_power(h, dimensions) / (n - 1);
    int query_block = checked_block_size(block_size);

    if (kernel->id == KERNEL_ID_GAUSSIAN) {
        double normalizer = 1.0;
        for (int d = 0; d < dimensions; d++) {
            normalizer *= MATHCONST_1_SQRT2PI;
        }
        if (!symmetric_self_gaussian(data, out, n, dimensions, inv_h,
                                     scale * normalizer, query_block)) {
            blocked_self_gaussian(data, out, n, dimensions, inv_h,
                                  scale * normalizer, query_block);
        }
        return;
    }

    switch (kernel->id) {
        case KERNEL_ID_EPANECHNIKOV:
            if (!symmetric_self_epanechnikov(data, out, n, dimensions, inv_h,
                                             scale, query_block)) {
                blocked_self_epanechnikov(data, out, n, dimensions, inv_h,
                                          scale, query_block);
            }
            return;
        case KERNEL_ID_TRIANGULAR:
            if (!symmetric_self_triangular(data, out, n, dimensions, inv_h,
                                           scale, query_block)) {
                blocked_self_triangular(data, out, n, dimensions, inv_h,
                                        scale, query_block);
            }
            return;
        case KERNEL_ID_UNIFORM:
            if (!symmetric_self_uniform(data, out, n, dimensions, inv_h,
                                        scale, query_block)) {
                blocked_self_uniform(data, out, n, dimensions, inv_h,
                                     scale, query_block);
            }
            return;
        case KERNEL_ID_COSINE:
            if (!symmetric_self_cosine(data, out, n, dimensions, inv_h,
                                       scale, query_block)) {
                blocked_self_cosine(data, out, n, dimensions, inv_h,
                                    scale, query_block);
            }
            return;
        case KERNEL_ID_GAUSSIAN:
            return;
    }
}
