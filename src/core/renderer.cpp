#include "core/io.h"
#include "core/renderer.h"
#include "core/texture.h"
#include "util/memory.h"

#include <glfw3.h>

Renderer::Renderer(const RendererProps &props, const std::string &optixir, SceneFacade &scene) : scene(scene), props(props)
{
    facade = std::make_unique<OptixFacade>(optixir, scene);

    glEnable(GL_TEXTURE_2D);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
}

void Renderer::Run(Window &window)
{
    std::unique_ptr<Texture> texture = std::make_unique<Texture>(window.GetWidth(), window.GetHeight());
    std::unique_ptr<IO> io = std::make_unique<IO>();

    size_t bytes = window.GetWidth() * window.GetHeight() * sizeof(float3);
    void *framebuffer_host = buffer_create_host(bytes);
    void *framebuffer_device = buffer_create_device(bytes);

    OptixParams parameters;
    parameters.seed = 0u;
    parameters.iteration = 0u;
    parameters.width = window.GetWidth();
    parameters.height = window.GetHeight();
    parameters.max_length = props.max_length;
    parameters.pixels = (float3 *)framebuffer_device;

    while (window)
    {
        scene.GetActiveCamera().CreateSnapshot(parameters.camera);
        facade->Launch(parameters);

        if ((parameters.iteration % props.iterations_to_refresh) == 0u)
        {
            buffer_copy_device_to_host(framebuffer_device, framebuffer_host, bytes);
            texture->Upload(framebuffer_host);

            glClear(GL_COLOR_BUFFER_BIT);
            glBegin(GL_TRIANGLES);
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(-1.0f, +1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(-1.0f, +1.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(+1.0f, +1.0f);
            glEnd();

            window.SwapBuffers();

            fprintf(stdout, "Iteration %u done.\n", parameters.iteration);
        }

        parameters.iteration++;
        window.Process();
        io->Process(window, scene, parameters);
    }

    buffer_delete_device(framebuffer_device);
    buffer_delete_host(framebuffer_host);
}
