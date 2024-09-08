#include "optix/optix_defs.h"
#include "optix/optix_facade.h"
#include "optix/optix_sbt.h"
#include "util/memory.h"

#include <optix_stubs.h>

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RecordCamera = Record<OptixTraversableHandle>;
using RecordMiss = Record<float3>;
using RecordHit = Record<MeshSnapshot>;

OptixSbt::OptixSbt(const OptixFacade &facade, const SceneFacade &scene, OptixTraversableHandle traversable) : facade(facade)
{
    CreateRecordsCamera(traversable);
    CreateRecordsMiss(scene);
    CreateRecordsHit(scene);
    CreateSbt(scene);
}

OptixSbt::~OptixSbt()
{
    buffer_delete_host(buffer_host_records_camera);
    buffer_delete_host(buffer_host_records_miss);
    buffer_delete_host(buffer_host_records_hit);
    buffer_delete_device(buffer_device_records_camera);
    buffer_delete_device(buffer_device_records_miss);
    buffer_delete_device(buffer_device_records_hit);
}

void OptixSbt::CreateRecordsCamera(OptixTraversableHandle traversable)
{
    size_t bytes = facade.groups_camera.size() * sizeof(RecordCamera);

    buffer_host_records_camera = buffer_create_host(bytes);
    buffer_device_records_camera = buffer_create_device(bytes);

    RecordCamera *records = (RecordCamera *)buffer_host_records_camera;
    OPTIX_CHECK(optixSbtRecordPackHeader(facade.groups_camera[0], &records[0]));
    records[0].data = traversable;

    buffer_copy_host_to_device(buffer_host_records_camera, buffer_device_records_camera, bytes);
}

void OptixSbt::CreateRecordsMiss(const SceneFacade &scene)
{
    size_t bytes = facade.groups_miss.size() * sizeof(RecordMiss);

    buffer_host_records_miss = buffer_create_host(bytes);
    buffer_device_records_miss = buffer_create_device(bytes);

    RecordMiss *records = (RecordMiss *)buffer_host_records_miss;
    OPTIX_CHECK(optixSbtRecordPackHeader(facade.groups_miss[0], &records[0]));
    records[0].data = scene.GetSky();

    buffer_copy_host_to_device(buffer_host_records_miss, buffer_device_records_miss, bytes);
}

void OptixSbt::CreateRecordsHit(const SceneFacade &scene)
{
    size_t bytes = scene.GetMeshes().size() * sizeof(RecordHit);

    buffer_host_records_hit = buffer_create_host(bytes);
    buffer_device_records_hit = buffer_create_device(bytes);

    RecordHit *records = (RecordHit *)buffer_host_records_hit;

    for (size_t i = 0u; i < scene.GetMeshes().size(); i++)
    {
        const Mesh &mesh = scene.GetMeshes()[i];
        const Material &material = scene.GetMaterials()[mesh.GetMaterialIndex()];

        OPTIX_CHECK(optixSbtRecordPackHeader(facade.groups_hit[material.illum], &records[i]));
        records[i].data.vertex_count = mesh.GetVertexCount();
        records[i].data.vertices = (const Vertex *)facade.globals->GetVertexBuffers()[i];
        records[i].data.material = (const Material *)facade.globals->GetMaterials()[mesh.GetMaterialIndex()];
        records[i].data.cdf = (const float *)facade.globals->GetCDFs()[i];
        records[i].data.area = mesh.GetArea();
    }

    buffer_copy_host_to_device(buffer_host_records_hit, buffer_device_records_hit, bytes);
}

void OptixSbt::CreateSbt(const SceneFacade &scene)
{
    memset(&sbt, 0, sizeof(OptixShaderBindingTable));
    sbt.raygenRecord = (CUdeviceptr)buffer_device_records_camera;
    sbt.missRecordBase = (CUdeviceptr)buffer_device_records_miss;
    sbt.missRecordStrideInBytes = sizeof(RecordMiss);
    sbt.missRecordCount = facade.groups_miss.size();
    sbt.hitgroupRecordBase = (CUdeviceptr)buffer_device_records_hit;
    sbt.hitgroupRecordStrideInBytes = sizeof(RecordHit);
    sbt.hitgroupRecordCount = scene.GetMeshes().size();
}
