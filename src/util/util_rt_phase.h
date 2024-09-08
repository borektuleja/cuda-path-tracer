#ifndef _UTIL_RT_PHASE_H_
#define _UTIL_RT_PHASE_H_

#include "math/math_float3.h"

inline __host__ __device__ float isotropic()
{
    return InvPi4;
}

inline __host__ __device__ float henyey_greenstein(float cosine, float g)
{
    float a = 1.0f - POW2(g);
    float b = powf(1.0f + POW2(g) + 2.0f * g * cosine, 3.0f / 2.0f);
    return InvPi4 * (a / b);
}

#endif
