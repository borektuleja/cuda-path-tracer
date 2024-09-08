#include "optix/optix_bvh.h"
#include "optix/optix_defs.h"
#include "optix/optix_facade.h"
#include "util/memory.h"

#include <optix_stubs.h>

OptixBvh::OptixBvh(const OptixFacade &facade, const SceneFacade &scene) : facade(facade)
{
    CreateBvh(scene);
}

OptixBvh::~OptixBvh()
{
    buffer_delete_device(buffer_device_traversable);
}

void OptixBvh::CreateBvh(const SceneFacade &scene)
{
    const unsigned int flags[] = {OPTIX_GEOMETRY_FLAG_NONE};

    std::vector<OptixBuildInput> builds(scene.GetMeshes().size());

    for (size_t i = 0u; i < scene.GetMeshes().size(); i++)
    {
        const Mesh &mesh = scene.GetMeshes()[i];

        memset(&builds[i], 0, sizeof(OptixBuildInput));
        builds[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        builds[i].triangleArray.vertexBuffers = (const CUdeviceptr *)&facade.globals->GetVertexBuffers()[i];
        builds[i].triangleArray.numVertices = mesh.GetVertexCount();
        builds[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        builds[i].triangleArray.vertexStrideInBytes = sizeof(Vertex);
        builds[i].triangleArray.flags = flags;
        builds[i].triangleArray.numSbtRecords = 1u;
    }

    OptixAccelBuildOptions options;
    memset(&options, 0, sizeof(OptixAccelBuildOptions));
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes;
    memset(&sizes, 0, sizeof(OptixAccelBufferSizes));
    OPTIX_CHECK(optixAccelComputeMemoryUsage(facade.context, &options, builds.data(), builds.size(), &sizes));

    void *buffer_device_compacted = buffer_create_device(sizeof(size_t));

    OptixAccelEmitDesc property;
    memset(&property, 0, sizeof(OptixAccelEmitDesc));
    property.result = (CUdeviceptr)buffer_device_compacted;
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    void *buffer_device_temporary = buffer_create_device(sizes.tempSizeInBytes);
    void *buffer_device_output = buffer_create_device(sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(facade.context, facade.stream, &options, builds.data(), builds.size(), (CUdeviceptr)buffer_device_temporary, sizes.tempSizeInBytes, (CUdeviceptr)buffer_device_output, sizes.outputSizeInBytes, &traversable, &property, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t compacted_size;
    buffer_copy_device_to_host(buffer_device_compacted, &compacted_size, sizeof(size_t));

    buffer_device_traversable = buffer_create_device(compacted_size);
    OPTIX_CHECK(optixAccelCompact(facade.context, facade.stream, traversable, (CUdeviceptr)buffer_device_traversable, compacted_size, &traversable));
    CUDA_CHECK(cudaDeviceSynchronize());

    buffer_delete_device(buffer_device_compacted);
    buffer_delete_device(buffer_device_temporary);
    buffer_delete_device(buffer_device_output);
}
