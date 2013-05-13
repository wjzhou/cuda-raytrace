#ifndef util_cu_h__
#define util_cu_h__
#include "optix_world.h"
#define INV_PI 0.31830988618379067154f
#define INV_TWOPI 0.15915494309189533577f
#define INV_FOURPI 0.07957747154594766788f

__device__ __inline__ float DistanceSquared(optix::float3& p1, optix::float3& p2){
    optix::float3 diff=p2-p1;
    return diff.x*diff.x+diff.y*diff.y+diff.z*diff.z;
}

__device__ __inline__ float AbsDot(const float3 &v1, const float3 &v2) {
    return fabsf(optix::dot(v1, v2));
}

__device__ __inline__ bool isBlack(CudaSpectrum &sp){
    return sp.x==0.0f && sp.y==0.0f && sp.z==0.0f;
};

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);

#define DEBUG_KERNEL


#endif // util.cu_h__
