#ifndef _MATH_FLOAT3_STD_H_
#define _MATH_FLOAT3_STD_H_

#include "math/math_float3.h"

inline __host__ __device__ float3 exp(const float3 &v)
{
    return make_float3(expf(v.x), expf(v.y), expf(v.z));
}

inline __host__ __device__ float3 sqrt(const float3 &v)
{
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

#endif
