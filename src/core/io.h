#ifndef _IO_H_
#define _IO_H_

#include "core/window.h"
#include "optix/optix_params.h"
#include "scene/scene_facade.h"

class IO
{
public:
    void Process(const Window &window, SceneFacade &scene, OptixParams &parameters) const;
};

#endif
