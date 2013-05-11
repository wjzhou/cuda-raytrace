#include <optix_world.h>
#include "cudashape.cu.h"
using namespace optix;
rtDeclareVariable(float, radius, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

__device__ __inline__ bool Quadratic(float A, float B, float C, float *t0, float *t1) {
    // Find quadratic discriminant
    float discrim = B * B - 4.f * A * C;
    if (discrim < 0.) return false;
    float rootDiscrim = sqrtf(discrim);

    // Compute quadratic _t_ values
    float q;
    if (B < 0) q = -.5f * (B - rootDiscrim);
    else       q = -.5f * (B + rootDiscrim);
    *t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1){
        float temp=*t0;
        *t0=*t1;
        *t1=temp;
    }
    return true;
}

RT_PROGRAM void sphere_intersect(int primIdx)
{
    float A=dot(ray.direction, ray.direction);
    float B=2.f*dot(ray.direction, ray.origin);
    float C=dot(ray.origin, ray.origin)-radius*radius;
    float t0,t1;
    if (!Quadratic(A, B, C, &t0, &t1))
        return;

    bool check_second=true;
    
    if(rtPotentialIntersection(t0)){
        float3 phit=ray.origin+t0*ray.direction;
        if (phit.x == 0.f && phit.y == 0.f) phit.x = 1e-5f * radius;
        float phi = atan2f(phit.y, phit.x);
        if (phi < 0.) phi += 2.f*M_PI;
        float u=phi/(2.0f*M_PI);
        float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
        float v = (theta)/(M_PI);
        uv=make_float2(u, v);

        shading_normal = geometry_normal = phit/radius;
        dpdu=make_float3(-shading_normal.y, shading_normal.x, 0.f);
        dpdv=cross(shading_normal, dpdu);
        if(rtReportIntersection(0)){
            check_second=false;
        }
    }
    if(check_second) {
        if(rtPotentialIntersection(t1)){
            float3 phit=ray.origin+t1*ray.direction;
            if (phit.x == 0.f && phit.y == 0.f) phit.x = 1e-5f * radius;
            float phi = atan2f(phit.y, phit.x);
            if (phi < 0.) phi += 2.f*M_PI;
            float u=phi/(2.0f*M_PI);
            float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
            float v = (theta)/(M_PI);
            uv=make_float2(u, v);

            shading_normal = geometry_normal = phit/radius;
            dpdu=make_float3(-shading_normal.y, shading_normal.x, 0.f);
            dpdv=cross(shading_normal, dpdu);
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void sphere_bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = make_float3(-radius);
  aabb->m_max = make_float3(radius);
}
