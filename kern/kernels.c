#include "kernels.h"

#include <string.h>

static const kernel1d_info KERNELS[] = {
    /* name, fn, is_symmetric, id, cutoff in bandwidths [COEFFICIENT ONLY ] */
    {"gaussian",     kernel_gaussian,     1, KERNEL_ID_GAUSSIAN,     4.0}, //4.0*h etc.
    {"epanechnikov", kernel_epanechnikov, 1, KERNEL_ID_EPANECHNIKOV, 1.0},
    {"triangular",   kernel_triangular,   1, KERNEL_ID_TRIANGULAR,   1.0},
    {"uniform",      kernel_uniform,      1, KERNEL_ID_UNIFORM,      1.0},
    {"cosine",       kernel_cosine,       1, KERNEL_ID_COSINE,       1.0},
};

static const int N_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

const kernel1d_info *get_kernel(const char *name) {
    for (int i = 0; i < N_KERNELS; i++) {
        if (strcmp(name, KERNELS[i].name) == 0) {
            return &KERNELS[i];
        }
    }
    return NULL;
}

kernel1d_fn get_kernel_fn(const char *name) {
    const kernel1d_info *kernel = get_kernel(name);
    return kernel ? kernel->fn : NULL;
}

int get_kernel_is_symmetric(const char *name) {
    const kernel1d_info *kernel = get_kernel(name);
    return kernel ? kernel->is_symmetric : 0;
}

double get_kernel_default_cutoff(const char *name) {
    const kernel1d_info *kernel = get_kernel(name);
    return kernel ? kernel->default_cutoff : 0.0;
}
