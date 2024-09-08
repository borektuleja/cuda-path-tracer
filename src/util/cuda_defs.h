#ifndef _CUDA_DEFS_H_
#define _CUDA_DEFS_H_

#include <stdio.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                                                                          \
    {                                                                                                                                                             \
        cudaError_t result = call;                                                                                                                                \
        if (result != cudaSuccess)                                                                                                                                \
            fprintf(stderr, "[CUDA] Line: %d, command(%s) failed with code %d. Reported message was %s.\n", __LINE__, #call, result, cudaGetErrorString(result)); \
    }

#endif
