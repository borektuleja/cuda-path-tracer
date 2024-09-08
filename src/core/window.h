#ifndef _WINDOW_H_
#define _WINDOW_H_

struct WindowProps
{
    unsigned int width;
    unsigned int height;
    const char *title;
};

class Window
{
public:
    Window(const WindowProps &props);
    ~Window();

public:
    operator bool() const;

public:
    void Process();
    void SwapBuffers();

public:
    unsigned int GetWidth() const;
    unsigned int GetHeight() const;
    void *GetHandle() const;

private:
    unsigned int width;
    unsigned int height;
    void *handle;
};

#endif
