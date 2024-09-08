#ifndef _OPTIX_PARAMS_H_
#define _OPTIX_PARAMS_H_

#include "scene/scene_defs.h"

struct OptixParams
{
    CameraSnapshot camera;
    unsigned int seed;
    unsigned int iteration;
    unsigned int width;
    unsigned int height;
    unsigned int max_length;
    float3 *pixels;
};

#endif
