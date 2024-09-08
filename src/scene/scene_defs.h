#ifndef _SCENE_DEFS_H_
#define _SCENE_DEFS_H_

#include "math/math_float3.h"

struct Vertex
{
    float3 v;
    float3 n;
};

struct Material
{
    int illum;

    float ior;
    float g;

    float3 emission;
    float3 diffuse;
    float3 sigma_a;
    float3 sigma_s;
};

struct CameraSnapshot
{
    float fov;

    float3 eye;
    float3 basis[3];
};

struct MeshSnapshot
{
    unsigned int vertex_count;
    const Vertex *vertices;
    const Material *material;
    const float *cdf;
    float area;
};

#endif
