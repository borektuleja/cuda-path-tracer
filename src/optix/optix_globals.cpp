#include "optix/optix_globals.h"
#include "util/memory.h"

OptixGlobals::OptixGlobals(SceneFacade &scene)
{
    CreateVertexBuffers(scene);
    CreateMaterials(scene);
    CreateCDFs(scene);
}

OptixGlobals::~OptixGlobals()
{
    DeleteCDFs();
    DeleteMaterials();
    DeleteVertexBuffers();
}

const std::vector<void *> &OptixGlobals::GetVertexBuffers() const
{
    return buffers_device_vertices;
}

const std::vector<void *> &OptixGlobals::GetMaterials() const
{
    return buffers_device_material;
}

const std::vector<void *> &OptixGlobals::GetCDFs() const
{
    return buffers_device_cdf;
}

void OptixGlobals::CreateVertexBuffers(const SceneFacade &scene)
{
    buffers_device_vertices.resize(scene.GetMeshes().size());

    for (size_t i = 0u; i < scene.GetMeshes().size(); i++)
    {
        const Mesh &mesh = scene.GetMeshes()[i];

        size_t bytes = mesh.GetVertexCount() * sizeof(Vertex);
        buffers_device_vertices[i] = buffer_create_device(bytes);
        buffer_copy_host_to_device(mesh.GetVertices(), buffers_device_vertices[i], bytes);
    }
}

void OptixGlobals::CreateMaterials(const SceneFacade &scene)
{
    buffers_device_material.resize(scene.GetMaterials().size());

    for (size_t i = 0u; i < scene.GetMaterials().size(); i++)
    {
        size_t bytes = sizeof(Material);
        buffers_device_material[i] = buffer_create_device(bytes);
        buffer_copy_host_to_device(&scene.GetMaterials()[i], buffers_device_material[i], bytes);
    }
}

void OptixGlobals::CreateCDFs(const SceneFacade &scene)
{
    buffers_device_cdf.resize(scene.GetMeshes().size());

    for (size_t i = 0u; i < scene.GetMeshes().size(); i++)
    {
        const Mesh &mesh = scene.GetMeshes()[i];

        size_t bytes = (mesh.GetVertexCount() / 3u) * sizeof(float);
        buffers_device_cdf[i] = buffer_create_device(bytes);
        buffer_copy_host_to_device(mesh.GetCDF(), buffers_device_cdf[i], bytes);
    }
}

void OptixGlobals::DeleteVertexBuffers()
{
    for (void *buffer_device_vertices : buffers_device_vertices)
    {
        buffer_delete_device(buffer_device_vertices);
    }
}

void OptixGlobals::DeleteMaterials()
{
    for (void *buffer_device_material : buffers_device_material)
    {
        buffer_delete_device(buffer_device_material);
    }
}

void OptixGlobals::DeleteCDFs()
{
    for (void *buffer_device_cdf : buffers_device_cdf)
    {
        buffer_delete_device(buffer_device_cdf);
    }
}
