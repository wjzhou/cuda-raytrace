#ifndef photonmapping_h__
#define photonmapping_h__
//This file is share by both cuda code and host code
#include "optix_world.h"
#include "cuda.h"
#include "util/common.cu.h"
struct RayTracingRecord
{
    optix::float3 position;
    optix::float3 dpdu;
    optix::float3 dpdv;
    optix::float3 shadingNormal;
    optix::float3 geometryNormal;
    optix::uint   flags;

    float         radius2;          
    float         photon_count;
    optix::float3 direction;
    optix::float3 flux;
    float         accum_atten;
    MaterialType  material;
    CUdeviceptr   materialParameter;
};
#define RayTracingRecordFlageException 0x01
#define RayTracingRecordFlageMISS 0x02

enum PMRayType{PM_RayTracingType=0, PM_PhotonTracingType, PM_GatheringType, PM_ShadowRayType, PM_NUM_RAY_TYPE};
enum PMRayPass{PM_RayTracingPass=0, PM_PhotonTracingPass, PM_GatheringPass, PM_NUM_Pass};


struct CudaPhoton{
    unsigned int hasLeftChild:1;
    unsigned int splitAxis:2;
    unsigned int rightChild:29;

    optix::float3 p;
    CudaSpectrum alpha;
    optix::float3 wi;
};
const unsigned int PHOTON_MAX_RIGHT_CHILD=(1<<29);

__device__ __host__ __inline__ void setValid(CudaPhoton& photon)
{
   photon.hasLeftChild=0x1;
}

__device__ __host__ __inline__ void setInValid(CudaPhoton& photon)
{
    photon.hasLeftChild=0x0;
}

__device__ __host__ __inline__ bool isValid(const CudaPhoton& photon)
{
    return photon.hasLeftChild;
}


#endif // photonmapping_h__
