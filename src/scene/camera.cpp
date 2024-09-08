#include <string.h>

#include "scene/camera.h"
#include "math/math_float3_lin.h"

constexpr float3 up = {0.0f, 1.0f, 0.0f};

Camera::Camera(const float3 &eye, const float3 &target, float fov) : fov(fov), eye(eye)
{
    forward = norm(target - eye);

    phi = atan2f(forward.z, forward.x);
    theta = acosf(forward.y);

    CreateLocalCoordinateSystem();
}

void Camera::MoveForwards(float speed)
{
    eye += speed * forward;
}

void Camera::MoveBackwards(float speed)
{
    eye -= speed * forward;
}

void Camera::AdjustPhi(float radians)
{
    phi += radians;
    CreateLocalCoordinateSystem();
}

void Camera::AdjustTheta(float radians)
{
    theta = fminf(fmaxf(0.0f + InvPi4, theta + radians), Pi - InvPi4);
    CreateLocalCoordinateSystem();
}

void Camera::CreateSnapshot(CameraSnapshot &snapshot) const
{
    snapshot.fov = fov;
    snapshot.eye = eye;
    memcpy(snapshot.basis, basis, sizeof(basis));
}

void Camera::CreateLocalCoordinateSystem()
{
    forward = spherical(phi, theta);
    local_coordinate_system(-forward, up, basis);
}
