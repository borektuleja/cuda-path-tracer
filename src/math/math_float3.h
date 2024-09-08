#ifndef _MATH_FLOAT3_H_
#define _MATH_FLOAT3_H_

#include "math/math_defs.h"

inline __host__ __device__ float3 zero()
{
    return make_float3(0.0f, 0.0f, 0.0f);
}

inline __host__ __device__ float3 one()
{
    return make_float3(1.0f, 1.0f, 1.0f);
}

inline __host__ __device__ float3 operator-(const float3 &v)
{
    return make_float3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float a, const float3 &b)
{
    return b * a;
}

inline __host__ __device__ float3 operator/(const float3 &a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator*=(float3 &a, float b)
{
    return a = a * b;
}

inline __host__ __device__ float3 operator/=(float3 &a, float b)
{
    return a = a / b;
}

inline __host__ __device__ float3 operator+=(float3 &a, const float3 &b)
{
    return a = a + b;
}

inline __host__ __device__ float3 operator-=(float3 &a, const float3 &b)
{
    return a = a - b;
}

inline __host__ __device__ float3 operator*=(float3 &a, const float3 &b)
{
    return a = a * b;
}

inline __host__ __device__ float3 operator/=(float3 &a, const float3 &b)
{
    return a = a / b;
}

#endif
