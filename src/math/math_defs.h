#ifndef _MATH_DEFS_H_
#define _MATH_DEFS_H_

#define _USE_MATH_DEFINES

#include <math.h>

#include <cuda_runtime.h>

#define POW2(x) ((x) * (x))
#define POW3(x) ((x) * (x) * (x))
#define POW4(x) ((x) * (x) * (x) * (x))
#define POW5(x) ((x) * (x) * (x) * (x) * (x))

constexpr float Pi = 3.1415927f;
constexpr float Pi2 = 2.0f * Pi;
constexpr float Pi4 = 4.0f * Pi;
constexpr float InvPi = 1.0f / Pi;
constexpr float InvPi2 = 1.0f / Pi2;
constexpr float InvPi4 = 1.0f / Pi4;

#define DEGTORAD(x) ((x) * Pi / 180.0f)
#define RADTODEG(x) ((x) * 180.0f / Pi)

#endif
