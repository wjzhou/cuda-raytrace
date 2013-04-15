#include "util/camera/camera.cu.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

struct RtRayPayload
{
     optix::uint ray_depth;
};

rtBuffer<float3, 2>  bOutput;
rtDeclareVariable(rtObject, top_group, ,);
rtDeclareVariable(float, scene_epsilon, ,);
//rtCallableProgram(CudaRayDifferential, cameraRay, ());
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
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


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, geometry_normal, attribute geometry_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, uv, attribute uv, );//tex coordinate
rtDeclareVariable(float3, dpdu, attribute dpdu, );
rtDeclareVariable(float3, dpdv, attribute dpdv, );
rtDeclareVariable(float, t, rtIntersectionDistance, );
//rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);

RT_PROGRAM void simple_cloest_hit()
{

    float3 world_shading_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometry_normal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, geometry_normal));
    //float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometry_normal
    float costha=dot(world_shading_normal, ray.direction);
    if(costha>0.0f)
        bOutput[launchIndex]=make_float3(costha,0.0f,0.0f);
    else
        bOutput[launchIndex]=make_float3(0.2f,0.2f,0.2f);

    //bOutput[launchIndex]=t*ray.direction+ray.origin;


    //bOutput[launchIndex]=make_float3(1.0f, 0.0f, 0.0f);

}

RT_PROGRAM void simple_miss()
{
    bOutput[launchIndex]=make_float3(0.3f, 0.3f, 0.0f);
}


RT_PROGRAM void simple_exception()
{
  bOutput[launchIndex] = make_float3(1.0f, 1.0f, 0.0f);
  rtPrintExceptionDetails();
}


