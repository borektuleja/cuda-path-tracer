#ifndef _OPTIX_BVH_H_
#define _OPTIX_BVH_H_

#include "scene/scene_facade.h"

#include <optix.h>

class OptixFacade;

class OptixBvh
{
    friend class OptixFacade;

public:
    OptixBvh(const OptixFacade &facade, const SceneFacade &scene);
    ~OptixBvh();

private:
    void CreateBvh(const SceneFacade &scene);

private:
    const OptixFacade &facade;
    OptixTraversableHandle traversable;
    void *buffer_device_traversable;
};

#endif
