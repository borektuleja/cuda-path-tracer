#include <fstream>
#include <sstream>

#include "core/renderer.h"
#include "scene/scene_builder.h"

static void load_binary_file(const char *filename, std::string &output)
{
    std::ifstream file(filename, std::ios::binary);
    std::stringstream stream;

    if (file.is_open())
    {
        stream << file.rdbuf();
        output = stream.str();
    }
}

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        fprintf(stderr, "Usage: ./bssrdf <optixir> <scene-xml> <width> <height> <iterations-to-refresh> <max-path-length>");
        exit(EXIT_FAILURE);
    }

    std::string optixir;
    load_binary_file(argv[1], optixir);

    SceneBuilder builder;
    builder.LoadXML(argv[2]);
    std::unique_ptr<SceneFacade> scene = builder.Build();

    WindowProps windowProps;
    windowProps.width = atoi(argv[3]);
    windowProps.height = atoi(argv[4]);
    windowProps.title = "TUL0009";

    RendererProps rendererProps;
    rendererProps.iterations_to_refresh = std::max((unsigned int)atoi(argv[5]), 1u);
    rendererProps.max_length = std::max((unsigned int)atoi(argv[6]), 5u);

    std::unique_ptr<Window> window = std::make_unique<Window>(windowProps);
    std::unique_ptr<Renderer> renderer = std::make_unique<Renderer>(rendererProps, optixir, *scene);

    renderer->Run(*window);
    return 0;
}
