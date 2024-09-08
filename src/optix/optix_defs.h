#ifndef _OPTIX_DEFS_H_
#define _OPTIX_DEFS_H_

#include <stdio.h>

#include <optix_types.h>

#define OPTIX_CHECK(call)                                                                                      \
    {                                                                                                          \
        OptixResult result = call;                                                                             \
        if (result != OPTIX_SUCCESS)                                                                           \
            fprintf(stderr, "[OPTIX] Line: %d, command (%s) failed with code %d.\n", __LINE__, #call, result); \
    }

#endif
