#ifndef _UTIL_RT_FRSNL_H_
#define _UTIL_RT_FRSNL_H_

#include "math/math_float3_fun.h"

inline __host__ __device__ float fresnel(const float3 &i, const float3 &n, float eta1, float eta2)
{
    const float eta = eta1 / eta2;
    const float cosI = dot(-i, n);
    const float sinT2 = POW2(eta) * (1.0f - POW2(cosI));

    if (sinT2 <= 1.0f)
    {
        const float cosT = sqrtf(1.0f - sinT2);
        const float ro = (eta1 * cosI - eta2 * cosT) / (eta1 * cosI + eta2 * cosT);
        const float rp = (eta2 * cosI - eta1 * cosT) / (eta2 * cosI + eta1 * cosT);
        return (POW2(ro) + POW2(rp)) / 2.0f;
    }

    return 1.0f;
}

#endif
