#include "scene/scene_facade.h"

SceneFacade::SceneFacade() : camera_index(0u) {}

void SceneFacade::SetActiveCamera(unsigned long index)
{
    if (index < cameras.size())
    {
        camera_index = index;
    }
}

Camera &SceneFacade::GetActiveCamera()
{
    return cameras[camera_index];
}

const std::vector<Material> &SceneFacade::GetMaterials() const
{
    return materials;
}

const std::vector<Camera> &SceneFacade::GetCameras() const
{
    return cameras;
}

const std::vector<Mesh> &SceneFacade::GetMeshes() const
{
    return meshes;
}

const float3 &SceneFacade::GetSky() const
{
    return sky;
}
