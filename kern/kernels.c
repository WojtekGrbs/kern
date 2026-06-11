#include "kernels.h"

#include <string.h>

static const kernel1d_info KERNELS[] = {
    {"gaussian",     kernel_gaussian,     1, KERNEL_ID_GAUSSIAN},
    {"epanechnikov", kernel_epanechnikov, 1, KERNEL_ID_EPANECHNIKOV},
    {"triangular",   kernel_triangular,   1, KERNEL_ID_TRIANGULAR},
    {"uniform",      kernel_uniform,      1, KERNEL_ID_UNIFORM},
    {"cosine",       kernel_cosine,       1, KERNEL_ID_COSINE},
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
