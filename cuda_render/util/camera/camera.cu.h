#ifndef cudacamera_cu_h__
#define cudacamera_cu_h__
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
#endif // cudacamera_cu_h__