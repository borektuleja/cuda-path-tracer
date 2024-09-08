#ifndef _RENDERER_H_
#define _RENDERER_H_

#include "core/window.h"
#include "optix/optix_facade.h"

struct RendererProps
{
    unsigned int iterations_to_refresh;
    unsigned int max_length;
};

class Renderer
{
public:
    Renderer(const RendererProps &props, const std::string &optixir, SceneFacade &scene);

public:
    void Run(Window &window);

private:
    SceneFacade &scene;
    std::unique_ptr<OptixFacade> facade;
    RendererProps props;
};

#endif
