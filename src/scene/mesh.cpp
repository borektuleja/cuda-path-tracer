#include "scene/mesh.h"
#include "util/shapes.h"

Mesh::Mesh(const Vertex *vertices, unsigned long vertex_count, unsigned long material_index) : vertices(vertices), vertex_count(vertex_count), material_index(material_index)
{
    cdf.resize(vertex_count / 3u);

    area = 0.0f;

    for (size_t i = 0u; i < vertex_count; i += 3u)
    {
        const float3 &a = vertices[i + 0u].v;
        const float3 &b = vertices[i + 1u].v;
        const float3 &c = vertices[i + 2u].v;

        const float pdf = area_of_triangle(a, b, c);
        area += pdf;

        cdf[i / 3u] = area;
    }

    for (size_t i = 0u; i < vertex_count; i += 3u)
    {
        cdf[i / 3u] /= area;
    }
}

const Vertex *Mesh::GetVertices() const
{
    return vertices;
}

unsigned long Mesh::GetVertexCount() const
{
    return vertex_count;
}

unsigned long Mesh::GetMaterialIndex() const
{
    return material_index;
}

const float *Mesh::GetCDF() const
{
    return cdf.data();
}

float Mesh::GetArea() const
{
    return area;
}
