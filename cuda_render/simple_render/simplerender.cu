#include "util/camera/camera.cu.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "util/light/cudalight.cu.h"
#include "util/camera/camera.cu.h"
#include "util/material/cudamaterial.cu.h"
#include "util/shape/cudashape.cu.h"
using namespace optix;

struct RtRayPayload
{
     optix::uint ray_depth;
};

rtBuffer<float3, 2>  bOutput;
rtDeclareVariable(rtObject, top_group, ,);
rtDeclareVariable(float, scene_epsilon, ,);
//rtCallableProgram(CudaRayDifferential, cameraRay, ());
//rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtBuffer<CudaRayDifferential, 2> bRays;

RT_PROGRAM void simple_camera(){
    CudaRayDifferential rd=bRays[launchIndex];
    optix::Ray ray(rd.o, rd.d, 0, scene_epsilon);
    RtRayPayload pld;
    pld.ray_depth=0u;
    rtTrace(top_group, ray, pld);
}

__device__ __inline__ float3 WorldtoLocal(const float3& v,
    const float3& nn, const float3& sn, const float3& tn)
{
    return make_float3(dot(v, sn), dot(v,tn), dot(v,nn));
}

struct ShadowPRD
{
  float attenuation;
};

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t, rtIntersectionDistance, );
rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );

//rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);

RT_PROGRAM void simple_cloest_hit()
{
    CudaSpectrum L=black();
    L.x=L.y=L.z=0.0f;
    const float3 point=ray.origin+ray.direction*t;

    float3 world_shading_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometry_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, geometry_normal));
    
    int totalLight=lightSize();
    //rtPrintf("1 %f,%f,%f\n", L.x,L.y,L.z);
    for (int i=0; i<totalLight; ++i)
    {
        float3 uwi;
        float pdf;
        CudaSpectrum li=Sample_L(i, point, uwi, pdf);
        Ray shadowRay(point, uwi, 1, 0.001f, 1.0f-0.001f);
        ShadowPRD pld;
        pld.attenuation=1.0f;
        rtTrace(top_group, shadowRay, pld);
        float3 wi=normalize(uwi);
        float3 wo=normalize(-ray.direction);
        //if(pld.attenuation>0.f)
        L+=pld.attenuation*fabs(dot(world_shading_normal, wi))*f(wo, wi)*li;
        //rtPrintf("2 %f,%f,%f,%f\n", L.x,L.y,L.z, li.x);
    }

    bOutput[launchIndex]=L;
#ifdef DEBUG_KERNEL
    if(isPrint()){
        rtPrintf("\nDL:%f %f %f", L.x, L.y, L.z);
    }
#endif
}

RT_PROGRAM void simple_miss()
{
//bOutput[launchIndex]=make_float3(0.3f, 0.3f, 0.0f);
    bOutput[launchIndex]=make_float3(0.0f);
}



RT_PROGRAM void simple_shadow_any_hit()
{
    shadow_prd.attenuation=0.f;
    rtTerminateRay();
}


RT_PROGRAM void simple_exception()
{
  bOutput[launchIndex] = make_float3(1.0f, 1.0f, 0.0f);
  rtPrintExceptionDetails();
}


