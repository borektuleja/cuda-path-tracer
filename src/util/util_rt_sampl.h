#ifndef _UTIL_RT_SAMPL_H_
#define _UTIL_RT_SAMPL_H_

#include "math/math_float3_lin.h"

inline __host__ __device__ float3 cosine_sample_hemisphere(float e1, float e2)
{
    const float sinT = sqrtf(1.0f - e2);
    const float x = cosf(Pi2 * e1) * sinT;
    const float y = sinf(Pi2 * e1) * sinT;
    const float z = sqrtf(e2);
    return make_float3(x, y, z);
}

inline __host__ __device__ float cosine_sample_hemisphere_pdf(float cosine)
{
    return InvPi * cosine;
}

inline __host__ __device__ float3 uniform_sample_hemisphere(float e1, float e2)
{
    const float sinT = sqrtf(1.0f - POW2(e2));
    const float x = cosf(Pi2 * e1) * sinT;
    const float y = sinf(Pi2 * e1) * sinT;
    const float z = e2;
    return make_float3(x, y, z);
}

inline __host__ __device__ float uniform_sample_hemisphere_pdf()
{
    return InvPi2;
}

inline __host__ __device__ float3 uniform_sample_sphere(float e1, float e2)
{
    const float sinT = sqrtf(1.0f - POW2(2.0f * e2 - 1.0f));
    const float x = cosf(Pi2 * e1) * sinT;
    const float y = sinf(Pi2 * e1) * sinT;
    const float z = 2.0f * e2 - 1.0f;
    return make_float3(x, y, z);
}

inline __host__ __device__ float uniform_sample_sphere_pdf()
{
    return InvPi4;
}

inline __host__ __device__ float3 henyey_greenstein_sample(float g, float e1, float e2)
{
    const float cosT = (1.0f / (2.0f * g)) * (1.0f + POW2(g) - POW2((1.0f - POW2(g)) / (1.0f - g + 2.0f * g * e1)));
    const float sinT = 1.0f - POW2(cosT);
    const float x = cosf(Pi2 * e2) * sinT;
    const float y = sinf(Pi2 * e2) * sinT;
    const float z = cosT;
    return norm(make_float3(x, y, z));
}

inline __host__ __device__ float3 uniform_sample_triangle(float e1, float e2)
{
    const float w = sqrtf(e1);
    const float x = 1.0f - w;
    const float y = w * (1.0f - e2);
    const float z = w * e2;
    return make_float3(x, y, z);
}

inline __host__ __device__ float uniform_sample_triangle_pdf(float area)
{
    return 1.0f / area;
}

#endif
