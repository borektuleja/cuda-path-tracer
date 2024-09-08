#ifndef _OPTIX_FACADE_H_
#define _OPTIX_FACADE_H_

#include <string>
#include <memory>

#include "optix/optix_bvh.h"
#include "optix/optix_globals.h"
#include "optix/optix_sbt.h"
#include "optix/optix_params.h"

class OptixFacade
{
    friend class OptixSbt;
    friend class OptixBvh;

public:
    OptixFacade(const std::string &optixir, SceneFacade &scene);
    ~OptixFacade();

public:
    void Launch(const OptixParams &parameters) const;

private:
    void CreateCudaStream();
    void CreateDeviceContext();
    void CreatePipelineSettings();
    void CreateModule(const std::string &optixir);
    void CreateProgramGroups();
    void CreatePipeline();
    void CreateObjects(SceneFacade &scene);
    void CreateParameters();

private:
    CUstream stream;
    OptixDeviceContext context;
    OptixPipelineCompileOptions options_compile;
    OptixPipelineLinkOptions options_link;
    OptixModule module;
    OptixPipeline pipeline;
    std::vector<OptixProgramGroup> groups_camera;
    std::vector<OptixProgramGroup> groups_miss;
    std::vector<OptixProgramGroup> groups_hit;
    std::unique_ptr<OptixGlobals> globals;
    std::unique_ptr<OptixBvh> bvh;
    std::unique_ptr<OptixSbt> sbt;
    void *buffer_device_parameters;
};

#endif
