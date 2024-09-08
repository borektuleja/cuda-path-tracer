#ifndef _SHAPES_H_
#define _SHAPES_H_

#include "math/math_float3_fun.h"

inline __host__ __device__ float area_of_triangle(const float3 &a, const float3 &b, const float3 &c)
{
    const float3 u = b - a;
    const float3 v = c - a;
    const float3 n = cross(u, v);
    return 0.5f * len(n);
}

#endif
