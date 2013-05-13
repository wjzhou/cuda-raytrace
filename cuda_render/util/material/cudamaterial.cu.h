#ifndef cudamaterial_cu_h__
#define cudamaterial_cu_h__

#include <optix_world.h>
#include "../common.cu.h"
#include "../util.cu.h"
#include "../shape/cudashape.cu.h"
#include "optix_cuda_interop.h"
#include "cudamaterial.common.cu.h"
using namespace optix;
//rtDeclareVariable(CudaSpectrum, kd, ,);
rtDeclareVariable(CUdeviceptr, materialParameter,,);
rtDeclareVariable(MaterialType, materialType, ,);

//use a different name with the rtDeclareVariable. In case I forget to add the method, the compiler
//can emit a error
__device__ __inline__ CudaSpectrum f_Lambert(CUdeviceptr parameter,
                                             const optix::float3& wo, const optix::float3& wi)
{
    return (*((CudaSpectrum*)parameter))*INV_PI;
}

__device__ __inline__ CudaSpectrum f(MaterialType materialType, CUdeviceptr materialParameter, 
                                     const optix::float3& wo, const optix::float3& wi)
{
    switch(materialType){
    case MaterialType::MaterialTypeMatt:
        return f_Lambert(materialParameter, wo, wi);
    }

    return black();
}
/*
__device__ __inline__ CudaSpectrum f(const optix::float3& wo, const optix::float3& wi)
{
    return f(materialType, materialParameter, wo, wi);
}*/

__device__ __inline__ bool SameHemisphere(const float3 &wo, const float3 &wi)
{
    return wo.z*wi.z>0.0f;
}

__device__ __inline__ float AbsCosTheta(const float3 &w) { return fabsf(w.z); }

__device__ __inline__ float Pdf(const float3 &wo, const float3 &wi)  {
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * INV_PI : 0.f;
}

__device__ __inline__ void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy) {
    float r, theta;
    // Map uniform random numbers to $[-1,1]^2$
    float sx = 2 * u1 - 1;
    float sy = 2 * u2 - 1;

    // Map square to $(r,\theta)$

    // Handle degeneracy at the origin
    if (sx == 0.0 && sy == 0.0) {
        *dx = 0.0;
        *dy = 0.0;
        return;
    }
    if (sx >= -sy) {
        if (sx > sy) {
            // Handle first region of disk
            r = sx;
            if (sy > 0.0) theta = sy/r;
            else          theta = 8.0f + sy/r;
        }
        else {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx/r;
        }
    }
    else {
        if (sx <= sy) {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy/r;
        }
        else {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx/r;
        }
    }
    theta *= M_PI / 4.f;
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}

__device__ __inline__ float3 CosineSampleHemisphere(float u1, float u2) {
    float3 ret;
    ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
    ret.z = sqrtf(max(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
    return ret;
}

__device__ __inline__ float3 WorldToLocal(const float3 &v, const float3 &nn, const float3 &sn, const float3 &tn)
{
    return make_float3(dot(v, sn), dot(v, tn), dot(v, nn));
}
__device__ __inline__ float3 LocalToWorld(const float3 &v, const float3 &nn, const float3 &sn, const float3 &tn) 
{
    return make_float3(sn.x * v.x + tn.x * v.y + nn.x * v.z,
        sn.y * v.x + tn.y * v.y + nn.y * v.z,
        sn.z * v.x + tn.z * v.y + nn.z * v.z);
}

__device__ __inline__ CudaSpectrum Sample_f_Lambert(const float3 &wo, float3 *wi,
                        float u1, float u2, float *pdf)
{
    // Cosine-sample the hemisphere, flipping the direction if necessary
    *wi = CosineSampleHemisphere(u1, u2);
    if (wo.z < 0.) wi->z *= -1.f;
        *pdf = Pdf(wo, *wi);
#ifdef DEBUG_KERNEL
        if(isPrint()){
            rtPrintf("\nwo:%f %f %f wi:%f %f %f", wo.x, wo.y, wo.z, wi->x, wi->y, wi->z);
        }
#endif
     return f_Lambert(materialParameter, wo, *wi);
}

__device__ __inline__ CudaSpectrum Sample_f(const float3 &wow, float3 *wiw,
                                            float u1, float u2, float *pdf)
{
    const float3 nn=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, aShadingNormal));
    const float3 sn=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, aDpdu));
    const float3 tn=cross(nn, sn);
  /*  if(isPrint()){
        rtPrintf("\nSn:%f %f %f, tn:%f %f %f, nn:%f %f %f u1: %f, u2:%f", sn.x, sn.y, sn.z, tn.x, tn.y, tn.z,nn.x, nn.y, nn.z,u1, u2);

    }
    */

    float3 wo=WorldToLocal(wow, nn, sn, tn);
    float3 wi;
    CudaSpectrum result=Sample_f_Lambert(wo, &wi, u1, u2, pdf);
    *wiw=LocalToWorld(wi, nn, sn, tn);
    return result;
}


__inline__ __device__ CudaSpectrum materialSpecularMirror(const float3 &wo, float3 *wi, bool photonRay)
{    
    *wi=make_float3(-wo.x, -wo.y, wo.z);
    return make_float3(1.f);
}

__inline__ __device__ float CosTheta(const float3 &w) { return w.z; }
__inline__ __device__ float SinTheta2(const float3 &w) {
    return max(0.f, 1.f - CosTheta(w)*CosTheta(w));
}

__inline__ __device__ CudaSpectrum materialSpecularGlass(const float3 &wo, float3 *wi, bool photonRay)
{    
    //bool entering=photonRay^(CosTheta(wo)>0.f);
    bool entering=CosTheta(wo)>0.f;
    float sini2=SinTheta2(wo);
    float eta=1.5f;
    if(entering){
        eta=1/1.5f;
    }else{
        eta=1.5f;
    }
    float sint2=eta*eta*sini2;
    if(sint2>=1.0f){ //total internal reflection
        return black();
    }
    float cost=sqrtf(fmax(0.f, 1.f-sint2));
    if (entering){
        cost=-cost;
    }
    float sintOverSini=eta;
    *wi=make_float3(sintOverSini*-wo.x, sintOverSini*-wo.y, cost);
    return make_float3(1.f);
}

__inline__ __device__ CudaSpectrum materialSpecular(const float3 &wow, float3 *wiw, bool photonRay, float3 p)
{
    const float3 nn=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, aShadingNormal));
    const float3 sn=normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, aDpdu));
    const float3 tn=cross(nn, sn);

    float3 wo=WorldToLocal(wow, nn, sn, tn);
    float3 wi;
    CudaSpectrum result=black();

    switch(materialType){
    case MaterialTypeMirror:
        result=materialSpecularMirror(wo, &wi, photonRay);
        break;
    case MaterialTypeGlass:
        result=materialSpecularGlass(wo, &wi, photonRay);
        break;
        //result=materialSpecularGlass(wo, &wi, photonRay);
    }
    
    *wiw=LocalToWorld(wi, nn, sn, tn);
    
    //float3 n=faceforward(nn, wow, nn);
    //*wiw=-wow+2.f*dot(nn, wow)*nn;
    
    //rtPrintf("\nwow: %f, %f, %f, nn: %f, %f, %f, \nwi: %f, %f, %f sn:%f, %f, %f", wow.x, wow.y, wow.z, nn.x, nn.y, nn.z, wi.x, wi.y, wi.z, sn.x,sn.y,sn.z);
    //rtPrintf("\nwo: %f, %f, %f, wi: %f, %f, %f", wo.x, wo.y, wo.z, wi.x, wi.y, wi.z);
    //rtPrintf("%d",materialType);
    return result;
}


__inline__ __device__ bool isSpecular(MaterialType material){
    if (material==MaterialTypeGlass||material==MaterialTypeMirror){
        return true;
    }
    return false;
}

#endif // cudamaterial.cu_h__
