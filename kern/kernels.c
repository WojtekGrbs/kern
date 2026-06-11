#include "kernels.h"

#include <string.h>

kernel1d_fn get_kernel_fn(const char *name) {
    if (strcmp(name, "gaussian")     == 0) return kernel_gaussian;
    if (strcmp(name, "epanechnikov") == 0) return kernel_epanechnikov;
    if (strcmp(name, "triangular")   == 0) return kernel_triangular;
    if (strcmp(name, "uniform")      == 0) return kernel_uniform;
    if (strcmp(name, "cosine")       == 0) return kernel_cosine;
    return NULL;
}
