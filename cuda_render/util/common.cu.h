#ifndef common_cu_h__
#define common_cu_h__

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

struct CudaRayDifferential{
    optix::float3 o;   // ray original
    optix::float3 d;   // ray direction
    optix::float3 rxOrigin; //ray original for the next pixel(scaled by multiply samples)
    optix::float3 ryOrigin;
    optix::float3 rxDirection;
    optix::float3 ryDirection;
};

typedef optix::float3 CudaSpectrum;
__device__ __inline__ CudaSpectrum black(){
    return optix::make_float3(0.f);
}

__device__ __inline__ CudaSpectrum CudaSpectrumFromFloat(float a){
    return optix::make_float3(a);
}
/*
struct CudaSpectrum{
    optix::float3 rgb;
    CudaSpectrum(float f){
        rgb.x=rgb.y=rgb.z=f;
    }
    CudaSpectrum operator*(const CudaSpectrum& other){
        CudaSpectrum result;
        result.rgb=rgb*other.rgb;
        return result;
    }
    CudaSpectrum& operator*=(const CudaSpectrum& other){
        rgb*=other.rgb;
        return *this;
    }
};
CudaSpectrum operator*(const CudaSpectrum& a, const CudaSpectrum& b){
    CudaSpectrum result;
    result.rgb=a.rgb*b.rgb;
    return result;
}
*/

struct CudaLightDevice{
    enum LightType{INVALID, POINT, DIRECTION, AREA_DISK};
    optix::float3 o; //the usage of o and p1,p2 is light type dependent
    LightType lt;
    optix::float3 p1;
    int nSample;
    optix::float3 p2;
    int randomStart;
    CudaSpectrum intensity;
    //int random2DStart;
    optix::float3 normal;
};

enum MaterialType{
    MaterialTypeMatt, MaterialTypeMirror, MaterialTypeGlass
};

#ifdef __CUDAC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#endif

