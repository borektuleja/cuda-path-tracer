#ifndef _SCENE_BUILDER_H_
#define _SCENE_BUILDER_H_

#include <memory>

#include "scene/scene_facade.h"

class SceneBuilder
{
public:
    SceneBuilder &AddCamera(const float3 &eye, const float3 &target, float fov = 0.785f);
    SceneBuilder &AddMesh(const char *filename);
    SceneBuilder &LoadXML(const char *filename);
    std::unique_ptr<SceneFacade> Build();

private:
    std::vector<VertexBuffer> vertex_buffers;
    std::vector<Material> materials;
    std::vector<Camera> cameras;
    std::vector<Mesh> meshes;
    float3 sky;
};

#endif
