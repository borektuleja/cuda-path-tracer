#ifndef _MESH_H_
#define _MESH_H_

#include <vector>

#include "scene/scene_defs.h"

class Mesh
{
public:
    Mesh(const Vertex *vertices, unsigned long vertex_count, unsigned long material_index);

public:
    const Vertex *GetVertices() const;
    unsigned long GetVertexCount() const;
    unsigned long GetMaterialIndex() const;
    const float *GetCDF() const;
    float GetArea() const;

private:
    const Vertex *vertices;
    unsigned long vertex_count;
    unsigned long material_index;
    std::vector<float> cdf;
    float area;
};

#endif
