#ifndef _MATH_FLOAT3_FUN_H_
#define _MATH_FLOAT3_FUN_H_

#include "math/math_float3.h"

inline __host__ __device__ float len(const float3 &v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float min(const float3 &v)
{
    return fminf(fminf(v.x, v.y), v.z);
}

inline __host__ __device__ float max(const float3 &v)
{
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

inline __host__ __device__ float chnl(const float3 &v, int i)
{
    return ((const float *)&v)[i];
}

inline __host__ __device__ float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 ortho(const float3 &v)
{
    return (fabsf(v.x) > fabsf(v.z)) ? make_float3(v.y, -v.x, 0.0f) : make_float3(0.0f, v.z, -v.y);
}

inline __host__ __device__ float3 norm(const float3 &v)
{
    return v / len(v);
}

inline __host__ __device__ float3 faceforward(const float3 &i, const float3 &n)
{
    return dot(i, n) < 0.0f ? -i : i;
}

inline __host__ __device__ float3 reflect(const float3 &i, const float3 &n)
{
    const float cosI = dot(-i, n);
    return i + 2.0f * cosI * n;
}

inline __host__ __device__ float3 refract(const float3 &i, const float3 &n, float eta)
{
    const float cosI = dot(-i, n);
    const float sinT2 = POW2(eta) * (1.0f - POW2(cosI));
    return (sinT2 > 1.0f) ? zero() : (eta * i + (eta * cosI - sqrtf(1.0f - sinT2)) * n);
}

inline __host__ __device__ float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

#endif
