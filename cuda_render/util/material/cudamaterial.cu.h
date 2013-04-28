#include <optix_world.h>
#include "../common.cu.h"
rtDeclareVariable(CudaSpectrum, kd, ,);
rtDeclareVariable(MaterialType, materialType, ,);

__device__ __inline__ CudaSpectrum f_Lambert(const optix::float3& wo, const optix::float3& wi)
{
    return kd; 
}


__device__ __inline__ CudaSpectrum f(const optix::float3& wo, const optix::float3& wi)
{
    switch(materialType){
    case MaterialType::MaterialTypeMatt:
        return f_Lambert(wo, wi);
    }
    return black();
}