#if defined(_MSC_VER)
#pragma once
#endif

#ifndef PhotonMappingRenderer_h__
#define PhotonMappingRenderer_h__

#include "scene.h"
#include "sampler.h"
#include "../cudarender.h"
#include <cuda.h>
#include "util/light/CudaLight.h"
#include "util/random/cudarandom.h"

class PhotonMappingRenderer : public CudaRenderer
{
public:
    PhotonMappingRenderer(Sampler *sampler, Camera *camera, const ParamSet &params);
    void render(const Scene *scene, CudaRender* cudarender, CudaCamera* cudacamera);
    
    ~PhotonMappingRenderer();
private:
    Sampler *sampler;
    Camera *camera;

    //inited in prelaunch
    const Scene *scene;
    CudaRender* cudarender;
    CudaCamera* cudacamera;

    CudaLight light;
    optix::Buffer bRayTracingOutput;

    RTsize width;
    RTsize height;

    void preLaunch();
    void RaytracingPass();
    void PhotonTracingPassPreLaunch();
    void PhotonTracingPass(int pass);
    void PhotonGatheringPass();
    void FinalGatheringPass();
    void postLaunch();
    void CreatePhotonMap(CUdeviceptr dIndirectPhotomap, unsigned int numPhotons);
    CUdeviceptr dIndirectPhotomap;
    CUdeviceptr dPhotonTracingRandom;
    CUdeviceptr dHaltonPermute;
    CudaRandom* rng;
    double totalPhotons;
};
#endif // PhotonMappingRenderer_h__