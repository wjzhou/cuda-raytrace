#ifndef cudatexture_cu_h__
#define cudatexture_cu_h__
//This is not used yet. It will be enhanced to support image texture and 
//procedure texture, such as noise. It is just a placeholder now.
#include "optix_world.h"
#include "../common.cu.h"
__inline__ __device__ CudaSpectrum TextureSpecture(float u, float v){
    return CudaSpectrumFromFloat(0.5f);
}

#endif // cudatexture.cu_h__
