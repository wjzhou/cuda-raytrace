#ifndef cudalight_cu_h__
#define cudalight_cu_h__
#include "../common.cu.h"
#include "../util.cu.h"
using namespace optix;
rtBuffer<CudaLightDevice, 1>  bLights;
rtBuffer<float4, 1>  bLightsAux;
__device__ __inline__ int lightSize(){
    return bLights.size();
}
rtBuffer<float, 3>  bLightRandom1D;
rtBuffer<optix::float2, 3>  bLightRandom2D;

__device__ __inline__ CudaSpectrum Sample_L_Point(const CudaLightDevice& cld, const float3& point,
    float3& uwi, float& pdf)
{
    uwi=cld.o-point; //unnormalized wi
        
    //rtPrintf("\ncld:%f, %f, %f, poing%f, %f, %f\n", cld.o.x, cld.o.y, cld.o.z, point.x, point.y, point.z );
    
    float invlength2=1.0f/dot(uwi, uwi);
    //wi=uwi*sqrtf(invlength2);
    pdf=1.f;
    return cld.intensity*invlength2;
}

__device__ __inline__ CudaSpectrum Sample_L_Area_Disk(const CudaLightDevice& cld, const float3& point,
    float3& wi, float& pdf)
{
    optix::uint3 index=make_uint3(cld.random2DStart, launchIndex);
    optix::float2 u=bLightRandom2D[index];
    float3 uwi=cld.o+u.x*cld.p1+u.y*cld.p2-point;
    
    wi=normalize(uwi);
    //pdf=1.f;
    float distanceSquared=dot(uwi,uwi);
    float costha=-dot(cld.normal, wi);
    //rtPrintf("point%f, %f, %f, cld.normal:%f,%f, %f, wi: %f, %f, %f",
    //    point.x, point.y, point.z, cld.normal.x, cld.normal.y, cld.normal.z, wi.x, wi.y, wi.z);
    pdf=distanceSquared/(costha*cld.area);
    return costha>0.0f? cld.intensity: black();
}

__device__ __inline__ CudaSpectrum Sample_L(const int iLight, const float3& point,
                                            float3& wi, float& pdf){
    CudaLightDevice cld=bLights[iLight];
    switch(cld.lt){
    case CudaLightDevice::POINT:
        return Sample_L_Point(cld, point, wi, pdf);
    case CudaLightDevice::AREA_DISK:
        //return Sample_L_Area_Disk(cld, point, wi, pdf);
        return Sample_L_Area_Disk(cld, point, wi, pdf);
    }
    return black();
}

__device__ __inline__ float3 UniformSampleSphere(float u1, float u2) {
    float z = 1.f - 2.f * u1;
    float r = sqrtf(max(0.f, 1.f - z*z));
    float phi = 2.f * M_PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, z);
}

__device__ __inline__ float UniformSpherePdf() {
    return 1.f / (4.f * M_PI);
}
__device__ __inline__ CudaSpectrum Sample_L_Point(const int iLight, float lu1, float lu2,
                                            float u1, float u2, Ray *ray, float3 *Ns, float *pdf)
{
    CudaLightDevice cld=bLights[iLight];
    ray->origin=cld.o;
    ray->direction=UniformSampleSphere(lu1, lu2);
    ray->tmax=0.f;
    *Ns = ray->direction;
    *pdf = UniformSpherePdf();
    return cld.intensity;
}

__device__ __inline__ CudaSpectrum Sample_L_Area_Disk(const int iLight, float lu1, float lu2,
                                                  float u1, float u2, Ray *ray, float3 *Ns, float *pdf)
{
    CudaLightDevice cld=bLights[iLight];

    float3 org = cld.o+lu1*cld.p1+lu2*cld.p2;
    float3 dir = UniformSampleSphere(u1, u2);
    *Ns=cld.normal;

    if (dot(dir, *Ns) < 0.) dir *= -1.f;
    ray->origin=org;
    ray->direction=dir;
    ray->tmin= 1e-3f;
    ray->tmax=RT_DEFAULT_MAX;

    *pdf=INV_TWOPI;
    return cld.intensity*cld.area; //should be add to the pdf
}


__device__ __inline__ CudaSpectrum Sample_L(const int iLight, float lu1, float lu2,
                                            float u1, float u2, Ray *ray, float3 *Ns, float *pdf){
    CudaLightDevice cld=bLights[iLight];
    switch(cld.lt){
    case CudaLightDevice::POINT:
        return Sample_L_Point(iLight, lu1, lu2, u1, u2, ray, Ns, pdf);
    case CudaLightDevice::AREA_DISK:
        return Sample_L_Area_Disk(iLight, lu1, lu2, u1, u2, ray, Ns, pdf);
    }
    
    return black();
}

//Use light L to prevent name collision with L local variable use in other place
//This is another feature of C++ that I do not like
__device__ __inline__ CudaSpectrum lightL(int iLight, const float3 &wow)
{
    //hit a light source
    if(iLight>=0){
        CudaLightDevice cld=bLights[iLight];
        if(dot(cld.normal, wow) > 0.f){
            return cld.intensity;
        }
    }
    return black();
}

#endif // cudalight_cu_h__
