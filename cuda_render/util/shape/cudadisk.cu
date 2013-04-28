#include <optix_world.h>
using namespace optix;
#define INV_TWOPI  0.15915494309189533577f

//input
rtDeclareVariable(float, height, attribute height, );
rtDeclareVariable(float, radius, attribute radius, );
rtDeclareVariable(float, innerRadius, attribute innerRadius, );
rtDeclareVariable(float, phiMax, attribute phiMax, );

//output
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, geometry_normal, attribute geometry_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, uv, attribute uv, );//tex coordinate
//rtDeclareVariable(float3, p, attribute p, ); //point position
rtDeclareVariable(float3, dpdu, attribute dpdu, );
rtDeclareVariable(float3, dpdv, attribute dpdv, );

RT_PROGRAM void disk_intersect(int primIdx)
{
    //optix::Ray ray; // to fool the vx assist..
    // Already in object space
    // Compute plane intersection for disk
    if (fabsf(ray.direction.z) < 1e-7) return;
    float thit = (height - ray.origin.z) / ray.direction.z;

    if(rtPotentialIntersection(thit)){
    // See if hit point is inside disk radii and $\phimax$
        float3 phit = ray.origin + thit*ray.direction;
        float dist2 = phit.x * phit.x + phit.y * phit.y;
        if (dist2 > radius * radius || dist2 < innerRadius * innerRadius)
            return;

    // Test disk $\phi$ value against $\phimax$
        float phi = atan2f(phit.y, phit.x);
        if (phi < 0) phi += 2.f * M_PI;
          if (phi > phiMax) return;

    // Find parametric representation of disk hit
        float oneMinusV = ((sqrtf(dist2)-innerRadius) /
                       (radius-innerRadius));
        float invOneMinusV = (oneMinusV > 0.f) ? (1.f / oneMinusV) : 0.f;
        uv=make_float2(phi/phiMax, 1.f-oneMinusV);
        dpdu=make_float3(-phiMax * phit.y, phiMax * phit.x, 0.f);
        dpdu *= phiMax * INV_TWOPI;
        rtTransformVector(RT_OBJECT_TO_WORLD, dpdu);
        dpdv=make_float3(-phit.x * invOneMinusV, -phit.y * invOneMinusV, 0.f);
        dpdv *= (radius - innerRadius) / radius;
        rtTransformVector(RT_OBJECT_TO_WORLD, dpdv);

        //I do not use the rayEpsilon currently, may use it in future
        //*rayEpsilon = 5e-4f * *tHit;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void disk_intersectP(int primIdx)
{
    if (fabsf(ray.direction.z) < 1e-7) return;
    float thit = (height - ray.origin.z) / ray.direction.z;
    if (rtPotentialIntersection(thit)){

    // See if hit point is inside disk radii and $\phimax$
    float3 phit = ray.origin + thit*ray.direction;
    float dist2 = phit.x * phit.x + phit.y * phit.y;
    if (dist2 > radius * radius || dist2 < innerRadius * innerRadius)
        return;

    // Test disk $\phi$ value against $\phimax$
    float phi = atan2f(phit.y, phit.x);
    if (phi < 0) phi += 2. * M_PI;
    if (phi > phiMax) return;
    rtReportIntersection(0);
    }
}


RT_PROGRAM void disk_bounds (int primIdx, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
 
    aabb->m_min = make_float3(-radius, -radius, height);
    aabb->m_max = make_float3( radius,  radius, height);
 
}

struct SampleResult{
    float3 point;
    float3 normal;
} ;
/*RT_CALLABLE_PROGRAM SampleResult disk_sample(float u1, float u2 )
{
    Point p;
    ConcentricSampleDisk(u1, u2, &p.x, &p.y);
    p.x *= radius;
    p.y *= radius;
    p.z = height;
    *Ns = Normalize((*ObjectToWorld)(Normal(0,0,1)));
    if (ReverseOrientation) *Ns *= -1.f;
    return (*ObjectToWorld)(p);
    SampleResult result;
    return result;

}*/
