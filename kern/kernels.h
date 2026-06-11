#ifndef KDE_KERNELS_H
#define KDE_KERNELS_H

#include <math.h>

#define MATHCONST_1_SQRT2PI 0.3989422804014327 /* 1 / sqrt(2*pi) */
#define MATHCONST_PI_4      0.7853981633974483 /* pi/4 */

typedef double (*kernel1d_fn)(double);

typedef enum {
    KERNEL_ID_GAUSSIAN = 0,
    KERNEL_ID_EPANECHNIKOV,
    KERNEL_ID_TRIANGULAR,
    KERNEL_ID_UNIFORM,
    KERNEL_ID_COSINE
} kernel1d_id;

typedef struct {
    const char *name;
    kernel1d_fn fn;
    int is_symmetric;
    kernel1d_id id;
} kernel1d_info;

/* Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-0.5*u^2) */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_gaussian(double u) {
    return MATHCONST_1_SQRT2PI * exp(-0.5 * u * u);
}

/* Epanechnikov kernel: K(u) = 0.75*(1-u^2) for |u|<=1, else 0 */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_epanechnikov(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.75 * (1.0 - u * u);
}

/* Triangular kernel: K(u) = (1-|u|) for |u|<=1, else 0 */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_triangular(double u) {
    double au = fabs(u);
    if (au >= 1.0) return 0.0;
    return 1.0 - au;
}

/* Uniform kernel: K(u) = 0.5 for |u|<=1, else 0 */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_uniform(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.5;
}

/* Cosine kernel: K(u) = (pi/4)*cos(pi/2 * u) for |u|<=1, else 0 */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_cosine(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return MATHCONST_PI_4 * cos(MATHCONST_PI_4 * 2.0 * u);
}

/* lgamma-based log-beta for numerical stability */
static inline double log_beta(double a, double b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/* Beta kernel. Evaluated in log-space to avoid overflow. */
static inline double kernel_beta(double x, double xi, double h) {
    if (x <= 0.0 || x >= 1.0) return 0.0;

    double a = xi / h + 1.0;
    double b = (1.0 - xi) / h + 1.0;
    double log_k = (a - 1.0) * log(x)
                 + (b - 1.0) * log(1.0 - x)
                 - log_beta(a, b);

    return exp(log_k);
}

const kernel1d_info *get_kernel(const char *name);
kernel1d_fn get_kernel_fn(const char *name);
int get_kernel_is_symmetric(const char *name);

#endif
