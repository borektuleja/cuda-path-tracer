#include <random>

#include "core/io.h"

#include <glfw3.h>

static std::random_device device;
static std::mt19937 generator(device());
static std::uniform_int_distribution<unsigned int> distribution;

void IO::Process(const Window &window, SceneFacade &scene, OptixParams &parameters) const
{
    GLFWwindow *handle = (GLFWwindow *)window.GetHandle();

    if (glfwGetKey(handle, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(handle, GLFW_TRUE);
    }

    for (unsigned int i = 0u; i <= 9u; i++)
    {
        if (glfwGetKey(handle, GLFW_KEY_0 + i) == GLFW_PRESS)
        {
            scene.SetActiveCamera(i);
            parameters.seed = distribution(generator);
            parameters.iteration = 0u;
        }
    }
}
