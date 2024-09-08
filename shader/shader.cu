#include <stdio.h>

#include "math/math_float3_lin.h"
#include "math/math_float3_std.h"
#include "optix/optix_params.h"
#include "util/algo.h"
#include "util/random.h"
#include "util/shapes.h"
#include "util/util_rt_frsnl.h"
#include "util/util_rt_phase.h"
#include "util/util_rt_sampl.h"

#include <optix_device.h>

enum
{
    PATH_FLAG_NONE = 0,
    PATH_FLAG_TERMINATE = (1 << 0),
    PATH_FLAG_MEDIUM = (1 << 1)
};

struct PathVertex
{
    unsigned int seed;

    float3 xo;
    float3 wo;

    float3 throughput;
    float3 radiance;

    unsigned int flags;
};

extern "C" __constant__ OptixParams params;

static __forceinline__ __device__ void pack(const void *address, unsigned int &p1, unsigned int &p2)
{
    uint64_t pointer = reinterpret_cast<uint64_t>(address);
    p1 = pointer >> 32;
    p2 = pointer & 0x00000000FFFFFFFF;
}

static __forceinline__ __device__ void *unpack(unsigned int p1, unsigned int p2)
{
    uint64_t pointer = static_cast<uint64_t>(p1) << 32 | p2;
    return reinterpret_cast<void *>(pointer);
}

static __forceinline__ __device__ float3 compute_v()
{
    return optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
}

static __forceinline__ __device__ float3 compute_n(const Vertex *vertices, int i = -1, bool flip = true)
{
    const unsigned int index = (i == -1) ? optixGetPrimitiveIndex() : i;

    const float3 &a = vertices[3u * index + 0u].n;
    const float3 &b = vertices[3u * index + 1u].n;
    const float3 &c = vertices[3u * index + 2u].n;

    const float2 uv = optixGetTriangleBarycentrics();
    const float3 n = (1.0f - uv.x - uv.y) * a + uv.x * b + uv.y * c;

    return flip ? faceforward(n, -optixGetWorldRayDirection()) : n;
}

extern "C" __global__ void __raygen__entry()
{
    const unsigned int x = optixGetLaunchIndex().x;
    const unsigned int y = optixGetLaunchIndex().y;
    const unsigned int i = x + y * params.width;

    OptixTraversableHandle traversable = *(OptixTraversableHandle *)optixGetSbtDataPointer();

    PathVertex vertex;
    unsigned int p1, p2;
    pack(&vertex, p1, p2);

    vertex.seed = tea<4u>(i, params.seed + params.iteration);

    float3 ndc;
    ndc.x = (x + rng(vertex.seed)) - params.width / 2.0f;
    ndc.y = (y + rng(vertex.seed)) - params.height / 2.0f;
    ndc.z = (params.height / (2.0f * tanf(params.camera.fov / 2.0f))) * (-1.0f);

    vertex.xo = params.camera.eye;
    vertex.wo = translate(norm(ndc), params.camera.basis);
    vertex.throughput = one();
    vertex.radiance = zero();
    vertex.flags = PATH_FLAG_NONE;

    for (unsigned int i = 0u; i < params.max_length; i++)
    {
        optixTrace(traversable, vertex.xo, vertex.wo, 1e-4f, 1e16f, 0.0f, OptixVisibilityMask(255u), OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p1, p2);

        if (vertex.flags & PATH_FLAG_TERMINATE)
            break;

        if (max(vertex.throughput) <= rng(vertex.seed))
            break;

        vertex.throughput *= 1.0f / max(vertex.throughput);
    }

    params.pixels[i] = (params.pixels[i] * params.iteration + vertex.radiance) / (params.iteration + 1.0f);
}

extern "C" __global__ void __miss__entry()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    float3 sky = *(const float3 *)optixGetSbtDataPointer();

    vertex.radiance = vertex.throughput * sky;
    vertex.flags = PATH_FLAG_TERMINATE;
}

extern "C" __global__ void __closesthit__entry_illum0()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    vertex.radiance = vertex.throughput * mesh.material->emission;
    vertex.flags = PATH_FLAG_TERMINATE;
}

extern "C" __global__ void __closesthit__entry_illum1()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    const float3 n = compute_n(mesh.vertices);

    float3 basis[3];
    local_coordinate_system(n, basis);

    const float3 wi = cosine_sample_hemisphere(rng(vertex.seed), 1.0f - rng(vertex.seed));

    vertex.xo = compute_v();
    vertex.wo = translate(wi, basis);
    vertex.throughput *= mesh.material->diffuse;
}

extern "C" __global__ void __closesthit__entry_illum2()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    const float3 n = compute_n(mesh.vertices);

    vertex.xo = compute_v();
    vertex.wo = reflect(vertex.wo, n);
    vertex.throughput *= mesh.material->diffuse;
}

extern "C" __global__ void __closesthit__entry_illum3()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    const float t = optixGetRayTmax();

    const float3 n = compute_n(mesh.vertices);

    const float eta1 = (vertex.flags & PATH_FLAG_MEDIUM) ? mesh.material->ior : 1.0f;
    const float eta2 = (vertex.flags & PATH_FLAG_MEDIUM) ? 1.0f : mesh.material->ior;

    vertex.xo = compute_v();

    vertex.throughput *= (vertex.flags & PATH_FLAG_MEDIUM) ? exp(-mesh.material->sigma_a * t) : one();

    if (rng(vertex.seed) < fresnel(vertex.wo, n, eta1, eta2))
    {
        vertex.wo = reflect(vertex.wo, n);
    }
    else
    {
        vertex.wo = refract(vertex.wo, n, eta1 / eta2);
        vertex.flags ^= PATH_FLAG_MEDIUM;
    }
}

extern "C" __global__ void __closesthit__entry_illum4()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    const float3 n = compute_n(mesh.vertices);

    const float eta1 = (vertex.flags & PATH_FLAG_MEDIUM) ? mesh.material->ior : 1.0f;
    const float eta2 = (vertex.flags & PATH_FLAG_MEDIUM) ? 1.0f : mesh.material->ior;

    if (vertex.flags & PATH_FLAG_MEDIUM)
    {
        const float3 sigma_a = mesh.material->sigma_a;
        const float3 sigma_s = mesh.material->sigma_s;
        const float3 sigma_t = sigma_a + sigma_s;

        const int i = (int)(3.0f * rng(vertex.seed));

        const float t = optixGetRayTmax();
        const float s = -logf(1.0f - rng(vertex.seed)) / chnl(sigma_t, i);

        vertex.xo += vertex.wo * fminf(t, s);

        if (s < t)
        {
            const float3 tau = exp(-sigma_t * s);

            float average = 0.0f;
            average += chnl(sigma_t, 0) * expf(-chnl(sigma_t, 0) * s);
            average += chnl(sigma_t, 1) * expf(-chnl(sigma_t, 1) * s);
            average += chnl(sigma_t, 2) * expf(-chnl(sigma_t, 2) * s);
            average /= 3.0f;

            float3 basis[3u];
            local_coordinate_system(vertex.wo, basis);

            const bool isotropic = fabsf(mesh.material->g) < 0.001f;
            const float e1 = rng(vertex.seed);
            const float e2 = rng(vertex.seed);
            const float3 wi = isotropic ? uniform_sample_sphere(e1, e2) : henyey_greenstein_sample(mesh.material->g, e1, e2);

            vertex.wo = translate(wi, basis);
            vertex.throughput *= tau * sigma_s / average;
        }
        else
        {
            const float3 tau = exp(-sigma_t * t);

            float average = 0.0f;
            average += expf(-chnl(sigma_t, 0) * t);
            average += expf(-chnl(sigma_t, 1) * t);
            average += expf(-chnl(sigma_t, 2) * t);
            average /= 3.0f;

            if (rng(vertex.seed) < fresnel(vertex.wo, n, eta1, eta2))
            {
                vertex.wo = reflect(vertex.wo, n);
            }
            else
            {
                vertex.wo = refract(vertex.wo, n, eta1 / eta2);
                vertex.flags ^= PATH_FLAG_MEDIUM;
            }

            vertex.throughput *= tau / average;
        }
    }
    else
    {
        vertex.xo = compute_v();

        if (rng(vertex.seed) < fresnel(vertex.wo, n, eta1, eta2))
        {
            vertex.wo = reflect(vertex.wo, n);
        }
        else
        {
            vertex.wo = refract(vertex.wo, n, eta1 / eta2);
            vertex.flags ^= PATH_FLAG_MEDIUM;
        }
    }
}

extern "C" __global__ void __closesthit__entry_illum5()
{
    const unsigned int p1 = optixGetPayload_0();
    const unsigned int p2 = optixGetPayload_1();
    PathVertex &vertex = *(PathVertex *)unpack(p1, p2);

    const MeshSnapshot &mesh = *(const MeshSnapshot *)optixGetSbtDataPointer();

    const float3 n = compute_n(mesh.vertices);

    const float eta1 = 1.0f;
    const float eta2 = mesh.material->ior;

    if (rng(vertex.seed) < fresnel(vertex.wo, n, eta1, eta2))
    {
        vertex.xo = compute_v();
        vertex.wo = reflect(vertex.wo, n);
    }
    else
    {
        const float3 sigma_a = mesh.material->sigma_a;
        const float3 sigma_s = mesh.material->sigma_s;
        const float3 sigma_t = sigma_a + sigma_s;

        const unsigned int i = binary_search_cdf(rng(vertex.seed), mesh.cdf, mesh.vertex_count / 3u);

        const float3 &a = mesh.vertices[3u * i + 0u].v;
        const float3 &b = mesh.vertices[3u * i + 1u].v;
        const float3 &c = mesh.vertices[3u * i + 2u].v;

        const float3 w = uniform_sample_triangle(rng(vertex.seed), rng(vertex.seed));

        const float3 xo = compute_v();
        const float3 xi = a * w.x + b * w.y + c * w.z;

        const float3 ni = compute_n(mesh.vertices, i, false);

        float3 basis[3];
        local_coordinate_system(ni, basis);

        const float Fdr = (-1.440f) / POW2(eta2) + 0.710 / eta2 + 0.668 + 0.0636 * eta2;
        const float A = (1.0f + Fdr) / (1.0f - Fdr);
        const float r = dot(xi - xo, xi - xo);
        const float3 sigma_tr = sqrt(3.0f * sigma_t * sigma_a);
        const float3 albedo = sigma_s / sigma_t;
        const float3 zr = one() / sigma_t;
        const float3 zv = zr + (4.0f / 3.0f) * A * zr;
        const float3 dr = sqrt(make_float3(r, r, r) + zr * zr);
        const float3 dv = sqrt(make_float3(r, r, r) + zv * zv);
        const float3 c1 = zr * (sigma_tr + (one() / dr));
        const float3 c2 = zv * (sigma_tr + (one() / dv));
        const float3 cr = c1 * exp(-sigma_tr * dr) / (dr * dr);
        const float3 cv = c2 * exp(-sigma_tr * dv) / (dv * dv);
        const float3 rd = albedo * InvPi4 * (cr + cv);

        const float3 wi = uniform_sample_hemisphere(rng(vertex.seed), 1.0f - rng(vertex.seed));

        float pdf = 1.0f;
        pdf *= 1.0f / mesh.area;
        pdf *= uniform_sample_hemisphere_pdf();

        vertex.xo = xi;
        vertex.wo = translate(wi, basis);
        vertex.throughput *= (rd * InvPi) / pdf;
    }
}

extern "C" __global__ void __anyhit__entry()
{
}
