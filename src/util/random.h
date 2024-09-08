#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cuda_runtime.h>

template <unsigned int N>
inline __host__ __device__ unsigned int tea(unsigned int v0, unsigned int v1)
{
    unsigned int y = v0;
    unsigned int z = v1;
    unsigned int sum = 0u;

    for (unsigned int n = 0u; n < N; n++)
    {
        sum += 0x9E3779B9;
        y += ((z << 4) + 0xA341316C) ^ (z + sum) ^ ((z >> 5) + 0xC8013EA4);
        z += ((y << 4) + 0xAD90777D) ^ (y + sum) ^ ((y >> 5) + 0x7E95761E);
    }

    return y;
}

inline __host__ __device__ float rng(unsigned int &seed)
{
    seed = seed * 1664525u + 1013904223u;
    return float(seed >> 8) / float(0x01000000u);
}

#endif
