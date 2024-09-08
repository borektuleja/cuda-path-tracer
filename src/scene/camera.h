#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "scene/scene_defs.h"

class Camera
{
public:
    Camera(const float3 &eye, const float3 &target, float fov);

public:
    void MoveForwards(float speed);
    void MoveBackwards(float speed);
    void AdjustPhi(float radians);
    void AdjustTheta(float radians);

public:
    void CreateSnapshot(CameraSnapshot &snapshot) const;

private:
    void CreateLocalCoordinateSystem();

private:
    float fov;
    float phi;
    float theta;
    float3 eye;
    float3 forward;
    float3 basis[3];
};

#endif
