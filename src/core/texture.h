#ifndef _TEXTURE_H_
#define _TEXTURE_H_

class Texture
{
public:
    Texture(unsigned int width, unsigned int height);
    ~Texture();

public:
    void Upload(const void *pixels);

private:
    unsigned int width;
    unsigned int height;
    unsigned int handle;
};

#endif
