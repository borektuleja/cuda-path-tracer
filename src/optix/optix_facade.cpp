#include "optix/optix_defs.h"
#include "optix/optix_facade.h"
#include "util/memory.h"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

static const char *entries[] =
{
    "__closesthit__entry_illum0",
    "__closesthit__entry_illum1",
    "__closesthit__entry_illum2",
    "__closesthit__entry_illum3",
    "__closesthit__entry_illum4",
    "__closesthit__entry_illum5"
};

static void callback(unsigned int level, const char *tag, const char *message, void *cbdata)
{
    fprintf(stderr, "[OPTIX] Log: %s.\n", message);
}

OptixFacade::OptixFacade(const std::string &optixir, SceneFacade &scene)
{
    CreateCudaStream();
    CreateDeviceContext();
    CreatePipelineSettings();
    CreateModule(optixir);
    CreateProgramGroups();
    CreatePipeline();
    CreateObjects(scene);
    CreateParameters();
}

OptixFacade::~OptixFacade()
{
    buffer_delete_device(buffer_device_parameters);
    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void OptixFacade::Launch(const OptixParams &parameters) const
{
    buffer_copy_host_to_device(&parameters, buffer_device_parameters, sizeof(OptixParams));
    OPTIX_CHECK(optixLaunch(pipeline, stream, (CUdeviceptr)buffer_device_parameters, sizeof(OptixParams), &sbt->sbt, parameters.width, parameters.height, 1u));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void OptixFacade::CreateCudaStream()
{
    CUDA_CHECK(cudaFree(NULL));
    CUDA_CHECK(cudaStreamCreate(&stream));
}

void OptixFacade::CreateDeviceContext()
{
    OptixDeviceContextOptions options;
    memset(&options, 0, sizeof(OptixDeviceContextOptions));
    options.logCallbackFunction = callback;
    options.logCallbackData = NULL;
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
}

void OptixFacade::CreatePipelineSettings()
{
    memset(&options_compile, 0, sizeof(OptixPipelineCompileOptions));
    options_compile.usesMotionBlur = 0;
    options_compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options_compile.numPayloadValues = 2;
    options_compile.numAttributeValues = 2;
    options_compile.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    options_compile.pipelineLaunchParamsVariableName = "params";
    options_compile.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    options_compile.allowOpacityMicromaps = 0;
    memset(&options_link, 0, sizeof(OptixPipelineLinkOptions));
    options_link.maxTraceDepth = 1u;
}

void OptixFacade::CreateModule(const std::string &optixir)
{
    OptixModuleCompileOptions options;
    memset(&options, 0, sizeof(OptixModuleCompileOptions));
    options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    options.boundValues = NULL;
    options.numBoundValues = 0u;
    options.numPayloadTypes = 0u;
    options.payloadTypes = NULL;
    OPTIX_CHECK(optixModuleCreate(context, &options, &options_compile, optixir.c_str(), optixir.size(), NULL, NULL, &module));
}

void OptixFacade::CreateProgramGroups()
{
    OptixProgramGroupOptions options;
    memset(&options, 0, sizeof(OptixProgramGroupOptions));
    options.payloadType = NULL;

    groups_camera.resize(1u);

    std::vector<OptixProgramGroupDesc> descriptors_camera(groups_camera.size());
    memset(&descriptors_camera[0], 0, sizeof(OptixProgramGroupDesc));
    descriptors_camera[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptors_camera[0].flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    descriptors_camera[0].raygen.module = module;
    descriptors_camera[0].raygen.entryFunctionName = "__raygen__entry";
    OPTIX_CHECK(optixProgramGroupCreate(context, descriptors_camera.data(), descriptors_camera.size(), &options, NULL, NULL, groups_camera.data()));

    groups_miss.resize(1u);

    std::vector<OptixProgramGroupDesc> descriptors_miss(groups_miss.size());
    memset(&descriptors_miss[0], 0, sizeof(OptixProgramGroupDesc));
    descriptors_miss[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptors_miss[0].flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    descriptors_miss[0].miss.module = module;
    descriptors_miss[0].miss.entryFunctionName = "__miss__entry";
    OPTIX_CHECK(optixProgramGroupCreate(context, descriptors_miss.data(), descriptors_miss.size(), &options, NULL, NULL, groups_miss.data()));

    groups_hit.resize(6u);

    std::vector<OptixProgramGroupDesc> descriptors_hit(groups_hit.size());

    for (size_t i = 0u; i < groups_hit.size(); i++)
    {
        memset(&descriptors_hit[i], 0, sizeof(OptixProgramGroupDesc));
        descriptors_hit[i].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptors_hit[i].flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        descriptors_hit[i].hitgroup.moduleCH = module;
        descriptors_hit[i].hitgroup.entryFunctionNameCH = entries[i];
        descriptors_hit[i].hitgroup.moduleAH = module;
        descriptors_hit[i].hitgroup.entryFunctionNameAH = "__anyhit__entry";
    }

    OPTIX_CHECK(optixProgramGroupCreate(context, descriptors_hit.data(), descriptors_hit.size(), &options, NULL, NULL, groups_hit.data()));
}

void OptixFacade::CreatePipeline()
{
    std::vector<OptixProgramGroup> groups;
    groups.insert(groups.end(), groups_camera.begin(), groups_camera.end());
    groups.insert(groups.end(), groups_miss.begin(), groups_miss.end());
    groups.insert(groups.end(), groups_hit.begin(), groups_hit.end());
    OPTIX_CHECK(optixPipelineCreate(context, &options_compile, &options_link, groups.data(), groups.size(), NULL, NULL, &pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 4u * 1024u, 4u * 1024u, 4u * 1024u, 1u));
}

void OptixFacade::CreateObjects(SceneFacade &scene)
{
    globals = std::make_unique<OptixGlobals>(scene);
    bvh = std::make_unique<OptixBvh>(*this, scene);
    sbt = std::make_unique<OptixSbt>(*this, scene, bvh->traversable);
}

void OptixFacade::CreateParameters()
{
    buffer_device_parameters = buffer_create_device(sizeof(OptixParams));
}
