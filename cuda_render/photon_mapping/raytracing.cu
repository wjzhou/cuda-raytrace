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

rtDeclareVariable(int, lightIndex, ,);
__device__ __inline__ CudaSpectrum directLight(const RayTracingRecord& rec){
    CudaSpectrum L=black();
    const float3& point=rec.position;

    float3 world_shading_normal = rec.shadingNormal;
    //    float3 world_geometry_normal = rec.geometryNormal;


    int totalLight=lightSize();

    if(lightIndex>=totalLight){
        rtPrintf("error, out of bound light index:%d", lightIndex);
        return L; //prevent GPU kernel crash
    }

    L+=lightL(lightIndex, -rec.direction);

    for (int i=0; i<totalLight; ++i)
    {
        int nSamples=lightNSamples(i);
        for (int iSample=0; iSample<nSamples; ++iSample){
            float3 uwi;
            float pdf;
            CudaSpectrum li=Sample_L(i, point, uwi, pdf, iSample);
            Ray shadowRay(point, uwi, PM_ShadowRayType, 0.001f, 1.0f-0.001f);
            ShadowPRD pld;
            pld.attenuation=1.0f;
            rtTrace(top_group, shadowRay, pld);
            float3 wi=normalize(uwi);
            float3 wo=normalize(-rec.direction);
            L+=pld.attenuation*fabs(dot(world_shading_normal, wi))*f(rec.material, rec.materialParameter, wo, wi)*li/(pdf*nSamples);
            //rtPrintf("attenuation:%f dot:%f lix:%f pdf:%f", pld.attenuation, fabs(dot(world_shading_normal, wi)), li.x, pdf);
        }
    }
    return L;
}

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

    RayTracingRecord& record=bRayTracingOutput[launchIndex];

    record.flags=0;
    record.position=point;
    record.dpdu=normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, aDpdu));
    record.dpdv=normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, aDpdv));
    record.shadingNormal=normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, aShadingNormal));
    record.geometryNormal=normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, aGeometryNormal));
    record.direction=ray.direction;
    record.material=materialType;
    record.materialParameter=materialParameter;
    record.flux=black();
    record.photon_count=0;
    record.radius2=4.f;
    CudaSpectrum DL=directLight(record);
    //CudaSpectrum DL=black();
    rtPrintf("\nDL: %f, %f, %f", DL.x, DL.y, DL.z);
    record.directLight=DL;
}

RT_PROGRAM void raytracing_miss()
{
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

