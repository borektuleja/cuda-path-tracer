#include "scene/scene_builder.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <pugixml.hpp>
#include <tiny_obj_loader.h>

SceneBuilder &SceneBuilder::AddCamera(const float3 &eye, const float3 &target, float fov)
{
    cameras.emplace_back(eye, target, fov);
    return *this;
}

SceneBuilder &SceneBuilder::AddMesh(const char *filename)
{
    tinyobj::ObjReader reader;

    if (reader.ParseFromFile(filename))
    {
        size_t material_offset = materials.size();

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        auto &mtls = reader.GetMaterials();

        for (const tinyobj::material_t &mtl : mtls)
        {
            Material &material = materials.emplace_back();
            memcpy(&material.illum, &mtl.illum, sizeof(int));
            memcpy(&material.ior, &mtl.ior, sizeof(float));
            memcpy(&material.g, &mtl.anisotropy, sizeof(float));
            memcpy(&material.sigma_a, &mtl.ambient, sizeof(float3));
            memcpy(&material.sigma_s, &mtl.specular, sizeof(float3));
            memcpy(&material.diffuse, &mtl.diffuse, sizeof(float3));
            memcpy(&material.emission, &mtl.emission, sizeof(float3));
        }

        for (const tinyobj::shape_t &shape : shapes)
        {
            VertexBuffer &vertex_buffer = vertex_buffers.emplace_back(shape.mesh.indices.size());

            for (size_t i = 0u; i < shape.mesh.indices.size(); i++)
            {
                const int vertex_index = shape.mesh.indices[i].vertex_index;
                const int normal_index = shape.mesh.indices[i].normal_index;
                memcpy(&vertex_buffer[i].v, &attrib.vertices[3u * vertex_index], sizeof(float3));
                memcpy(&vertex_buffer[i].n, &attrib.normals[3u * normal_index], sizeof(float3));
            }

            const Vertex *vertices = vertex_buffer.data();
            unsigned long vertex_count = vertex_buffer.size();
            unsigned long material_index = material_offset + shape.mesh.material_ids[0];
            Mesh &mesh = meshes.emplace_back(vertices, vertex_count, material_index);
        }
    }

    return *this;
}

SceneBuilder &SceneBuilder::LoadXML(const char *filename)
{
    pugi::xml_document document;
    pugi::xml_parse_result result = document.load_file(filename);

    if (result)
    {
        sky.x = document.child("Scene").child("Sky").attribute("r").as_float();
        sky.y = document.child("Scene").child("Sky").attribute("g").as_float();
        sky.z = document.child("Scene").child("Sky").attribute("b").as_float();

        for (pugi::xml_node element : document.child("Scene").child("Cameras"))
        {
            float3 eye;
            eye.x = element.child("Eye").attribute("x").as_float();
            eye.y = element.child("Eye").attribute("y").as_float();
            eye.z = element.child("Eye").attribute("z").as_float();

            float3 target;
            target.x = element.child("Target").attribute("x").as_float();
            target.y = element.child("Target").attribute("y").as_float();
            target.z = element.child("Target").attribute("z").as_float();

            AddCamera(eye, target, DEGTORAD(element.attribute("fov").as_float()));
        }

        for (pugi::xml_node element : document.child("Scene").child("Models"))
        {
            AddMesh(element.attribute("path").as_string());
        }
    }

    return *this;
}

std::unique_ptr<SceneFacade> SceneBuilder::Build()
{
    std::unique_ptr<SceneFacade> facade = std::make_unique<SceneFacade>();
    facade->vertex_buffers = std::move(vertex_buffers);
    facade->materials = std::move(materials);
    facade->cameras = std::move(cameras);
    facade->meshes = std::move(meshes);
    facade->sky = sky;
    return facade;
}
