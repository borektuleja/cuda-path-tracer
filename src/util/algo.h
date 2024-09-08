#ifndef _ALGO_H_
#define _ALGO_H_

#include <cuda_runtime.h>

inline __host__ __device__ int binary_search_cdf(float e, const float *cdf, size_t n)
{
    size_t l = 0u;
    size_t r = n - 1u;

    for (;;)
    {
        if ((l + 1u) == r)
        {
            if (e <= cdf[l])
                return l;
            else
                return r;
        }

        size_t c = l + (r - l) / 2u;

        if (e <= cdf[c])
            r = c;
        else
            l = c;
    }
}

#endif
