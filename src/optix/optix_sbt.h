#ifndef _OPTIX_SBT_H_
#define _OPTIX_SBT_H_

#include "scene/scene_facade.h"

#include <optix.h>

class OptixFacade;

class OptixSbt
{
    friend class OptixFacade;

public:
    OptixSbt(const OptixFacade &facade, const SceneFacade &scene, OptixTraversableHandle traversable);
    ~OptixSbt();

private:
    void CreateRecordsCamera(OptixTraversableHandle traversable);
    void CreateRecordsMiss(const SceneFacade &scene);
    void CreateRecordsHit(const SceneFacade &scene);
    void CreateSbt(const SceneFacade &scene);

private:
    const OptixFacade &facade;
    OptixShaderBindingTable sbt;
    void *buffer_host_records_camera;
    void *buffer_host_records_miss;
    void *buffer_host_records_hit;
    void *buffer_device_records_camera;
    void *buffer_device_records_miss;
    void *buffer_device_records_hit;
};

#endif
