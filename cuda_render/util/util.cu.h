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

#define DEBUG_KERNEL


#endif // util.cu_h__
