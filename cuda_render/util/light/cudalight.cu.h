#ifndef cudalight_cu_h__
#define cudalight_cu_h__
#include "../common.cu.h"
using namespace optix;
rtBuffer<CudaLightDevice, 1>  bLights;
rtBuffer<float4, 1>  bLightsAux;
__device__ __inline__ int lightSize(){
    return bLights.size();
}
rtBuffer<float, 3>  bRandom1D;
rtBuffer<optix::float2, 3>  bRandom2D;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);



__device__ __inline__ CudaSpectrum Sample_L_Point(const CudaLightDevice& cld, const float3& point,
    float3& uwi, float& pdf)
{
    uwi=cld.o-point; //unnormalized wi
    float invlength2=1.0f/dot(uwi, uwi);
    //wi=uwi*sqrtf(invlength2);
    pdf=1.f;
    return cld.intensity*invlength2;
}

__device__ __inline__ CudaSpectrum Sample_L_Area_Disk(const CudaLightDevice& cld, const float3& point,
    float3& wi, float& pdf)
{
    optix::uint3 index=make_uint3(cld.randomStart, launchIndex);
    optix::float2 u=bRandom2D[index];
    float3 uwi=cld.o+u.x*cld.p1+u.y*cld.p2-point;
    
    wi=normalize(uwi);
    //pdf=1.f;
    float distanceSquared=dot(uwi,uwi);
    
    pdf=distanceSquared/dot(cld.normal, wi);
    return cld.intensity;
}

__device__ __inline__ CudaSpectrum Sample_L(const int iLight, const float3& point,
                                            float3& wi, float& pdf){
    CudaLightDevice cld=bLights[iLight];
    switch(cld.lt){
    case CudaLightDevice::POINT:
        return Sample_L_Point(cld, point, wi, pdf);
    case CudaLightDevice::AREA_DISK:
        //rtPrintf("almost not possible:%d\n", iLight);
        return Sample_L_Area_Disk(cld, point, wi, pdf);
    }
    return black();
}

/*
__device__ __inline__ CudaSpectrum Sample_L_emit(const int iLight, const float3& point,
                                            Ray float& pdf){
    CudaLightDevice cld=bLights[iLight];
    switch(cld.lt){
        
        case CudaLightDevice::POINT:
        float3 uwi=cld.o-point; //unnormalized wi
        float3 invlength2=1.0f/dot(uwi, uwi);
        wi=uwi*sqrtf(invlength2)
        pdf=1.f;     
    return cld.intensity/=invlength2;
                                               
}*/

#endif // cudalight_cu_h__
