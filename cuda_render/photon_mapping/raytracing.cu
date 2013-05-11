#include "util/camera/camera.cu.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "util/light/cudalight.cu.h"
#include "util/camera/camera.cu.h"
#include "util/material/cudamaterial.cu.h"
#include "photonmapping.h"
using namespace optix;

struct RtRayPayload
{
     optix::uint ray_depth;
};

rtDeclareVariable(rtObject, top_group, ,);
rtDeclareVariable(float, scene_epsilon, ,);
rtBuffer<CudaRayDifferential, 2> bRays;

RT_PROGRAM void raytracing_camera(){
    CudaRayDifferential rd=bRays[launchIndex];
    optix::Ray ray(rd.o, rd.d, PM_RayTracingType, scene_epsilon);
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
rtDeclareVariable(RtRayPayload, raytracingPayLoad, rtPayload, );
rtDeclareVariable(ShadowPRD, shadow_prd, rtPayload, );
//rtDeclareVariable(int, lightsource,,); //if the shape is not a light source
                                      // this is  -1, otherwise the index to light source

//rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
//rtDeclareVariable(MaterialType, materialType, ,);
rtBuffer<RayTracingRecord, 2> bRayTracingOutput;
RT_PROGRAM void raytracing_closest_hit()
{
    const float3 point=ray.origin+ray.direction*t;
    if (isSpecular(materialType)){
        Ray newRay;
        CudaSpectrum spec=materialSpecular(-ray.direction, &newRay.direction, true, point);
        newRay.origin=point;
        newRay.ray_type=PM_RayTracingType;
        newRay.tmin=scene_epsilon;
        newRay.tmax=RT_DEFAULT_MAX;
        raytracingPayLoad.ray_depth++;
        if (raytracingPayLoad.ray_depth>10){
            bRayTracingOutput[launchIndex].flags=RayTracingRecordFlageException;
            return;
        }
        rtTrace(top_group, newRay, raytracingPayLoad);
        return;
    }

    RayTracingRecord record;
    float3 world_shading_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometry_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, geometry_normal));
    
    record.flags=0;
    record.position=point;
    record.dpdu=dpdu;
    record.dpdv=dpdv;
    record.shadingNormal=shading_normal;
    record.geometryNormal=geometry_normal;
    record.direction=ray.direction;
    record.material=materialType;
    record.materialParameter=materialParameter;

    bRayTracingOutput[launchIndex]=record;
}

RT_PROGRAM void raytracing_miss()
{
//bOutput[launchIndex]=make_float3(0.3f, 0.3f, 0.0f);
    bRayTracingOutput[launchIndex].flags=RayTracingRecordFlageMISS;
}


RT_PROGRAM void raytracing_exception()
{
  bRayTracingOutput[launchIndex].flags=RayTracingRecordFlageException;
  rtPrintExceptionDetails();
}


RT_PROGRAM void shadow_any_hit()
{
    shadow_prd.attenuation=0.f;
    rtTerminateRay();
}

