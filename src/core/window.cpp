#include "core/window.h"

#include <glfw3.h>

Window::Window(const WindowProps &props) : width(props.width), height(props.height)
{
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    handle = glfwCreateWindow(props.width, props.height, props.title, NULL, NULL);
    glfwMakeContextCurrent((GLFWwindow *)handle);
}

Window::~Window()
{
    glfwDestroyWindow((GLFWwindow *)handle);
    glfwTerminate();
}

Window::operator bool() const
{
    return (glfwWindowShouldClose((GLFWwindow *)handle) == GLFW_FALSE);
}

void Window::Process()
{
    glfwPollEvents();
}

void Window::SwapBuffers()
{
    glfwSwapBuffers((GLFWwindow *)handle);
}

unsigned int Window::GetWidth() const
{
    return width;
}

unsigned int Window::GetHeight() const
{
    return height;
}

void *Window::GetHandle() const
{
    return handle;
}
