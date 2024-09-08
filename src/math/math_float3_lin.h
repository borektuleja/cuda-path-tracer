#ifndef _MATH_FLOAT3_LIN_H_
#define _MATH_FLOAT3_LIN_H_

#include "math/math_float3_fun.h"

inline __host__ __device__ void local_coordinate_system(const float3 &forward, float3 basis[3])
{
    basis[2] = norm(forward);
    basis[0] = norm(cross(ortho(forward), basis[2]));
    basis[1] = norm(cross(basis[2], basis[0]));
}

inline __host__ __device__ void local_coordinate_system(const float3 &forward, const float3 &up, float3 basis[3])
{
    basis[2] = norm(forward);
    basis[0] = norm(cross(up, basis[2]));
    basis[1] = norm(cross(basis[2], basis[0]));
}

inline __host__ __device__ float3 translate(const float3 &v, const float3 basis[3])
{
    return v.x * basis[0] + v.y * basis[1] + v.z * basis[2];
}

inline __host__ __device__ float3 spherical(float phi, float theta)
{
    return make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
}

#endif
