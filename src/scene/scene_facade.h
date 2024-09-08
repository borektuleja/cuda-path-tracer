#ifndef _SCENE_FACADE_H_
#define _SCENE_FACADE_H_

#include <vector>

#include "scene/camera.h"
#include "scene/mesh.h"

using VertexBuffer = std::vector<Vertex>;

class SceneBuilder;

class SceneFacade
{
    friend class SceneBuilder;

public:
    SceneFacade();

public:
    void SetActiveCamera(unsigned long index);

public:
    Camera &GetActiveCamera();
    const std::vector<Material> &GetMaterials() const;
    const std::vector<Camera> &GetCameras() const;
    const std::vector<Mesh> &GetMeshes() const;
    const float3 &GetSky() const;

private:
    unsigned long camera_index;
    std::vector<VertexBuffer> vertex_buffers;
    std::vector<Material> materials;
    std::vector<Camera> cameras;
    std::vector<Mesh> meshes;
    float3 sky;
};

#endif
