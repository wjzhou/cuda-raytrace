#include <optix_world.h>
#include "../common.cu.h"
#include "cuda.h"
//rtDeclareVariable(CudaSpectrum, kd, ,);
rtDeclareVariable(CUdeviceptr, materialParameter,,);
rtDeclareVariable(MaterialType, materialType, ,);



__device__ __inline__ CudaSpectrum f_Lambert(const optix::float3& wo, const optix::float3& wi)
{
    return *((CudaSpectrum*)materialParameter);
}


__device__ __inline__ CudaSpectrum f(const optix::float3& wo, const optix::float3& wi)
{
    switch(materialType){
    case MaterialType::MaterialTypeMatt:
        return f_Lambert(wo, wi);
    }
    return black();
}