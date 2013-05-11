
#include "cuda.h"
#include "util/cuda/helper_cuda.h" // Because the optix pullin some of the cuda runtime API but not all
                                   // include helper_cuda here to avoid problem
#include "PhotonMappingRenderer.h"
#include "film.h"
#include "../cudarender.h"
#include "util/util.h"
#include "util/camera/cudacamera.h"
#include "photonmapping.h"
#include "core/camera.h"
#include "util/light/cudalight.h"
#include "util/random/cudarandom.h"
#include "pbrt.h"
#include <../../../../Program Files (x86)/Microsoft Visual Studio 10.0/VC/include/stdio.h>

PhotonMappingRenderer::PhotonMappingRenderer(Sampler *s, Camera *c, const ParamSet &params)
{
    sampler=s;
    camera=c;
}

PhotonMappingRenderer::~PhotonMappingRenderer()
{
    delete sampler;
    delete camera;
}

void PhotonMappingRenderer::render(const Scene *ascene, CudaRender* acudarender, CudaCamera* acudacamera) {
    scene=ascene;
    cudarender=acudarender;
    cudacamera=acudacamera;
    preLaunch();
    RaytracingPass();
    PhotonTracingPass();
    GatheringPass();
    postLaunch();
}

void PhotonMappingRenderer::preLaunch(){

    gContext->setPrintEnabled(true);
    gContext->setPrintBufferSize(4096);

    gContext->setStackSize(2048);
    gContext->setRayTypeCount(PM_NUM_RAY_TYPE);
    gContext->setEntryPointCount(PM_NUM_Pass);
    gContext["scene_epsilon"]->setFloat(0.1f);
    optix::Material m=gContext->createMaterial();
    //load programs
    //ray tracing
    const string progRaytracingPTX=ptxpath("photon_mapping", "raytracing.cu");
    gContext->setMissProgram(PM_RayTracingPass, gContext->createProgramFromPTXFile(progRaytracingPTX,
        "raytracing_miss"));
    gContext->setRayGenerationProgram(PM_RayTracingPass, gContext->createProgramFromPTXFile(progRaytracingPTX,
        "raytracing_camera"));

    optix::Program progRayTracingClosestHit=gContext->createProgramFromPTXFile(progRaytracingPTX,
        "raytracing_closest_hit");
    m->setClosestHitProgram(PM_RayTracingType, progRayTracingClosestHit);
    optix::Program progShadowAnyHit=gContext->createProgramFromPTXFile(progRaytracingPTX,
        "shadow_any_hit");
    m->setAnyHitProgram(PM_ShadowRayType, progShadowAnyHit);

    //photon tracing
    const string progPhotonTracingPTX=ptxpath("photon_mapping", "photontracing.cu");
   // gContext->setMissProgram(PM_PhotonTracingPass, gContext->createProgramFromPTXFile(progPhotonTracingPTX,
   //     "photontracing_miss"));
    gContext->setRayGenerationProgram(PM_PhotonTracingPass, gContext->createProgramFromPTXFile(progPhotonTracingPTX,
        "photontracing_camera"));
    gContext->setExceptionProgram(PM_PhotonTracingPass, gContext->createProgramFromPTXFile(progPhotonTracingPTX,
        "photontracing_exception"));
    gContext->setMissProgram(PM_PhotonTracingPass, gContext->createProgramFromPTXFile(progPhotonTracingPTX,
        "photontracing_miss"));
    optix::Program progPhotonTracingClosestHit=gContext->createProgramFromPTXFile(progPhotonTracingPTX,
        "photontracing_closest_hit");
     m->setClosestHitProgram(PM_PhotonTracingType, progPhotonTracingClosestHit);
     m->setAnyHitProgram(PM_PhotonTracingType, gContext->createProgramFromPTXFile(progPhotonTracingPTX,
         "photontracing_debug_anyhit"));

    //gathering
    const string progGatheringPTX=ptxpath("photon_mapping", "gathering.cu");
    gContext->setRayGenerationProgram(PM_GatheringPass, gContext->createProgramFromPTXFile(progGatheringPTX,
        "gathering_camera"));
    
    for (std::vector<optix::GeometryInstance>::iterator
        it=cudarender->geometryInstances.begin();
        it!= cudarender->geometryInstances.end(); ++it)
    {
        (*it)->setMaterialCount(1);
        (*it)->setMaterial(0, m);
    }
    //dummy bOutput define, optix need this to pass validation, even I do not use it in the first two pass..
    optix::Buffer bOutput=gContext->createBuffer(RT_BUFFER_OUTPUT);
    bOutput->setFormat(RT_FORMAT_FLOAT3);
    bOutput->setSize(1, 1);
    gContext["bOutput"]->set(bOutput);
}

void PhotonMappingRenderer::RaytracingPass(){


    //Sample* sample=new Sample(sampler, nullptr, nullptr, scene);
    CudaSample* cudaSample=new CudaSample(sampler);
    light.preLaunch(scene, cudaSample);
    cudacamera->init(camera, sampler, cudaSample);
    
    cudacamera->getExtent(width, height);

    cudacamera->preLaunch(scene);
    bRayTracingOutput=gContext->createBuffer(RT_BUFFER_OUTPUT);
    bRayTracingOutput->setFormat(RT_FORMAT_USER);
    bRayTracingOutput->setElementSize(sizeof(RayTracingRecord));
    bRayTracingOutput->setSize(width, height);
    gContext["bRayTracingOutput"]->set(bRayTracingOutput);

    gContext->launch(PM_RayTracingPass,
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height)
        );

    /*RayTracingRecord* pRayTracingOutput= reinterpret_cast<RayTracingRecord*> (bRayTracingOutput->map());
    for(int i=0;i<width*height;++i){
        long long a=(long long)pRayTracingOutput[i].materialParameter;
        if (pRayTracingOutput[i].material!=0||a<1000000){
           // printf("%d",a);
        }
    }
    bRayTracingOutput->unmap();*/
}


struct Photon {
    Photon(const Point &pp, const CudaSpectrum &wt, const optix::float3 &w)
        : p(pp), alpha(wt), wi(w) { }
    Photon() { }
    Point p;
    CudaSpectrum alpha;
    optix::float3 wi;
};

void PhotonMappingRenderer::CreatePhotonMap(CUdeviceptr dPhotonMap, unsigned int numPhotons)
{
    CudaPhoton* hPhotonMap=new CudaPhoton[numPhotons];
    checkCudaErrors(cuMemcpyDtoH(hPhotonMap, dPhotonMap, numPhotons * sizeof(CudaPhoton)));
    std::vector<Photon> photons;
    photons.reserve(numPhotons);
    //unsigned int totalValidPhotons=0;
    for (unsigned int i=0; i<numPhotons; ++i){
        const CudaPhoton& photon=hPhotonMap[i];
        if(isValid(photon)){
            //emplace_back is not implemented in vs 2010
            photons.push_back(TPhoton(photon));
        }
    }
    Info("Valid photons:%d, Total photons:%d", photons.size(), numPhotons);
    if(photons.size()>0){
        KdTree<TPhoton> kdtree(photons);
        for (unsigned int i=0; i<kdtree.nNodes; ++i){
            hPhotonMap[i]=kdtree.nodeData[i].cudaphoton;
            hPhotonMap[i].splitAxis=kdtree.nodes[i].splitAxis;
            hPhotonMap[i].hasLeftChild=kdtree.nodes[i].hasLeftChild;
            hPhotonMap[i].rightChild=kdtree.nodes[i].rightChild;
        }

        checkCudaErrors(cuMemcpyHtoD(dPhotonMap, hPhotonMap, kdtree.nNodes * sizeof(CudaPhoton)));
    }
    delete[] hPhotonMap;
}

#define PhotonTracingRandomPerSample 3
#define PhotonTracingMaxDepth        4
#define PhotonTracingMaxWIDTH        512
#define PhotonTracingMaxHEIGHT       512
void PhotonMappingRenderer::PhotonTracingPass(){

    Info("Starting photon tracing pass...");
    const unsigned int PhotonTracingPhotons=PhotonTracingMaxWIDTH*PhotonTracingMaxHEIGHT*PhotonTracingMaxDepth;
    const unsigned int PhotonTracingSamples=PhotonTracingPhotons*PhotonTracingRandomPerSample;

    CUdeviceptr tPhotonTracingRandom; 
    checkCudaErrors(
        cuMemAlloc(&tPhotonTracingRandom, sizeof(float)*PhotonTracingSamples));
    gContext["photonTracingRandom"]->setUserData(sizeof(CUdeviceptr), &tPhotonTracingRandom);
    CudaRandom rng;

    CUdeviceptr tIndirectPhotomap;
    checkCudaErrors(
        cuMemAlloc(&tIndirectPhotomap, sizeof(CudaPhoton)*PhotonTracingPhotons));
    gContext["indirectPhotonmap"]->setUserData(sizeof(CUdeviceptr), &tIndirectPhotomap);

    rng.generate((float*)tPhotonTracingRandom, PhotonTracingSamples);    
    gContext["lightSourceIndex"]->setUint(0);
    gContext["max_photon_count"]->setUint(PhotonTracingMaxDepth);
    gContext["photonTracinglaunchWidth"]->setUint(PhotonTracingMaxWIDTH);
    gContext["photonTracingEmittingPhotons"]->setUint(PhotonTracingMaxWIDTH*PhotonTracingMaxHEIGHT);
    
    gContext->launch(PM_PhotonTracingPass,
        PhotonTracingMaxWIDTH, PhotonTracingMaxHEIGHT);
    CreatePhotonMap(tIndirectPhotomap, PhotonTracingPhotons);

    //cuMemFree(tIndirectPhotomap);
    cuMemFree(tPhotonTracingRandom);
}

void PhotonMappingRenderer::GatheringPass()
{
    optix::Buffer bOutput=gContext->createBuffer(RT_BUFFER_OUTPUT);
    bOutput->setFormat(RT_FORMAT_FLOAT3);
    bOutput->setSize(width, height);
    gContext["bOutput"]->set(bOutput);
    gContext->launch(PM_GatheringPass,
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height)
        );

    optix::float3* pOutput= reinterpret_cast<optix::float3*> (bOutput->map());
    int totoalSample=width*height;
    CameraSample* csamples=cudacamera->getCameraSamples();

    //FILE *file = fopen("pm.txt","w");
    for(int i=0; i<totoalSample; ++i){
        RGBSpectrum result=RGBSpectrum::FromRGB(reinterpret_cast<float*>(&pOutput[i]));
        // Issue warning if unexpected radiance value returned
        if (result.HasNaNs()) {
            Error("Not-a-number radiance value returned "
                "for image sample.  Setting to black.");
            result = Spectrum(0.f);
        }
        else if (result.y() < -1e-5) {
            Error("Negative luminance value, %f, returned"
                "for image sample.  Setting to black.", result.y());
            result=Spectrum(0.f);
        }
        else if (isinf(result.y())) {
            Error("Infinite luminance value returned"
                "for image sample.  Setting to black.");
            result = Spectrum(0.f);
        }
        camera->film->AddSample(csamples[i],result);
        //fprintf(file, "\niX:%f\tiY%f\t", csamples[i].imageX, csamples[i].imageY);
        //result.Print(file);    
    }
   // fclose(file);
}

void PhotonMappingRenderer::postLaunch()
{
    light.postLaunch();
    cudacamera->postLaunch();
    camera->film->WriteImage();
}
