#ifndef _OPTIX_GLOBALS_H_
#define _OPTIX_GLOBALS_H_

#include "scene/scene_facade.h"

class OptixGlobals
{
public:
    OptixGlobals(SceneFacade &scene);
    ~OptixGlobals();

public:
    const std::vector<void *> &GetVertexBuffers() const;
    const std::vector<void *> &GetMaterials() const;
    const std::vector<void *> &GetCDFs() const;

private:
    void CreateVertexBuffers(const SceneFacade &scene);
    void CreateMaterials(const SceneFacade &scene);
    void CreateCDFs(const SceneFacade &scene);
    void DeleteVertexBuffers();
    void DeleteMaterials();
    void DeleteCDFs();

private:
    std::vector<void *> buffers_device_vertices;
    std::vector<void *> buffers_device_material;
    std::vector<void *> buffers_device_cdf;
};

#endif
