#include <optix_world.h>
#include "cudashape.cu.h"
using namespace optix;


//input
rtDeclareVariable(float2, invRadius2, , );
rtDeclareVariable(float, innerRadius, , );
rtDeclareVariable(float, phiMax, , );
rtDeclareVariable(float3, worldx, , );
rtDeclareVariable(float3, worldy, , );
rtDeclareVariable(float3, worldz, , );
rtDeclareVariable(float3, worldo, , );
rtDeclareVariable(float, moffset, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void disk_intersect(int primIdx)
{
    //divid 0 is ok, the IEEE 754 will get inf, and the if will fail
    float thit=(moffset-dot(worldz,ray.origin))/dot(worldz,ray.direction);
    if (thit>ray.tmin && thit<ray.tmax){
        float3 phit = ray.origin + thit*ray.direction;

        float3 localPhit=phit-worldo;
        float localx=dot(localPhit, worldx)*invRadius2.x;
        float localy=dot(localPhit, worldy)*invRadius2.y;

        float dist2=localx*localx+localy*localy;
        if (dist2 > 1.f || dist2 < innerRadius*innerRadius)
        return;

        float phi = atan2f(localy, localx);
        if (phi < 0) phi += 2.f * M_PI;
        if (phi > phiMax) return;
        if(rtPotentialIntersection(thit)){
           
            aShadingNormal=aGeometryNormal=worldz;
            float oneMinusV = ((sqrt(dist2)-innerRadius) /
                (1.f-innerRadius));
            //float invOneMinusV = (1.f / oneMinusV);
             aUv=make_float2(phi/phiMax, 1.f-oneMinusV);
            aDpdu = -localy*worldx+localx*worldy;
            //dpdu *= phiMax * INV_TWOPI;
            aDpdv = -localx*worldx-localy*worldy;
            //dpdv *= (radius - innerRadius) / radius;*/
            rtReportIntersection(0);
        }
    }    
}

/*    //optix::Ray ray; // to fool the vx assist..
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
}*/

RT_PROGRAM void disk_bounds (int primIdx, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    float3 p0=worldo+worldx+worldy;
    float3 p1=worldo+worldx-worldy;
    float3 p2=worldo-worldx+worldy;
    float3 p3=worldo-worldx-worldy;
    aabb->m_min = fminf(fminf(fminf(p0, p1), p2),p3);
    aabb->m_max = fmaxf(fmaxf(fmaxf(p0, p1), p2),p3);
}

