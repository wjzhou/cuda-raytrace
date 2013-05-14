
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
#include "montecarlo.h"

PhotonMappingRenderer::PhotonMappingRenderer(Sampler *s, Camera *c, const ParamSet &params)
{
    sampler=s;
    camera=c;
    totalPhotons=0.0;
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
    PhotonTracingPassPreLaunch();
    int passes=1;
    for (int i=0; i<passes; ++i){
        PhotonTracingPass(i);
        PhotonGatheringPass();
    }
    FinalGatheringPass();
    postLaunch();
}

void PhotonMappingRenderer::preLaunch(){

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
    gContext->setRayGenerationProgram(PM_PhotonGatheringPass, gContext->createProgramFromPTXFile(progGatheringPTX,
        "photonGatheringCamera"));
    gContext->setRayGenerationProgram(PM_FinalGatheringPass, gContext->createProgramFromPTXFile(progGatheringPTX,
        "finalGatheringCamera"));    


    
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
    
    cudacamera->init(camera, sampler, cudaSample);
    cudacamera->getExtent(width, height);
    light.preLaunch(scene, cudaSample, width, height);

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

//used by fall back pbrt kdtree build on CPU, correct is concern, performance not
struct TPhoton {
    TPhoton(const CudaPhoton& cudaphoton)
        : p(pointfromFloat3(cudaphoton.p)), cudaphoton(cudaphoton) { }
    TPhoton() { }
    Point p; //require by pbrt kdtree template
    CudaPhoton cudaphoton;
};

void PhotonMappingRenderer::CreatePhotonMap(CUdeviceptr dPhotonMap, unsigned int numPhotons)
{
    CudaPhoton* hPhotonMap=new CudaPhoton[numPhotons];
    checkCudaErrors(cuMemcpyDtoH(hPhotonMap, dPhotonMap, numPhotons * sizeof(CudaPhoton)));
    std::vector<TPhoton> photons;
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
    if(photons.size()==0){
        Severe("0 photon numbers, quit");
    }
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


void PhotonMappingRenderer::PhotonTracingPassPreLaunch(){
    const unsigned int PhotonTracingPhotons=PhotonTracingMaxWIDTH*PhotonTracingMaxHEIGHT*PhotonTracingMaxDepth;
    const unsigned int PhotonTracingSamples=PhotonTracingPhotons*PhotonTracingRandomPerSample;
    checkCudaErrors(
        cuMemAlloc(&dPhotonTracingRandom, sizeof(float)*PhotonTracingSamples));
    gContext["photonTracingRandom"]->setUserData(sizeof(CUdeviceptr), &dPhotonTracingRandom);
    checkCudaErrors(
        cuMemAlloc(&dIndirectPhotomap, sizeof(CudaPhoton)*PhotonTracingPhotons));
    gContext["indirectPhotonmap"]->setUserData(sizeof(CUdeviceptr), &dIndirectPhotomap);
    
    PermutedHalton hal(5, RNG());
    checkCudaErrors(cuMemAlloc(&dHaltonPermute, hal.sumBases*sizeof(unsigned int)));
    gContext["haltonPermute"]->setUserData(sizeof(CUdeviceptr), &dHaltonPermute);
    rng=new CudaRandom();

}
void PhotonMappingRenderer::PhotonTracingPass(int pass){

    Info("Starting photon tracing pass...");
    const unsigned int PhotonTracingPhotons=PhotonTracingMaxWIDTH*PhotonTracingMaxHEIGHT*PhotonTracingMaxDepth;
    const unsigned int PhotonTracingSamples=PhotonTracingPhotons*PhotonTracingRandomPerSample;

    rng->generate((float*)dPhotonTracingRandom, PhotonTracingSamples);    
    gContext["lightSourceIndex"]->setUint(0);
    gContext["max_photon_count"]->setUint(PhotonTracingMaxDepth);
    gContext["photonTracinglaunchWidth"]->setUint(PhotonTracingMaxWIDTH);
    totalPhotons+=(PhotonTracingMaxWIDTH*PhotonTracingMaxHEIGHT);

    PermutedHalton hal(5, RNG(pass++));
    checkCudaErrors(cuMemcpyHtoD(dHaltonPermute, hal.permute, hal.sumBases*sizeof(unsigned int)));

    
    gContext->launch(PM_PhotonTracingPass,
        PhotonTracingMaxWIDTH, PhotonTracingMaxHEIGHT);
    CreatePhotonMap(dIndirectPhotomap, PhotonTracingPhotons);

    //cuMemFree(tIndirectPhotomap);
    //cuMemFree(dPhotonTracingRandom);
}

void PhotonMappingRenderer::PhotonGatheringPass(){
    gContext->launch(PM_PhotonGatheringPass, 
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height));
}

void PhotonMappingRenderer::FinalGatheringPass()
{
    int totoalSample=width*height;
    optix::Buffer bOutput=gContext->createBuffer(RT_BUFFER_OUTPUT);
    bOutput->setFormat(RT_FORMAT_FLOAT3);
    bOutput->setSize(width, height);
    gContext["bOutput"]->set(bOutput);
    gContext["emittingPhotons"]->setFloat((float) totalPhotons);
    printf("preFInal");
    gContext->launch(PM_FinalGatheringPass,
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height));
     printf("postFInal");
    optix::float3* pOutput= reinterpret_cast<optix::float3*> (bOutput->map());
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
    bOutput->unmap();
    bOutput->destroy();
    //fclose(file);
    
}

void PhotonMappingRenderer::postLaunch()
{
    light.postLaunch();
    cudacamera->postLaunch();
    camera->film->WriteImage();
}
