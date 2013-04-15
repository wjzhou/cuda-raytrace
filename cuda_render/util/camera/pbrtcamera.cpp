#include "pbrtcamera.h"
#include "util/util.h"
#include "cudarender.h"
#include "camera.h"
#include "film.h"

void initCudaRDFromRD(CudaRayDifferential& dst, const RayDifferential& src){
    dst.o=float3fromPoint(src.o);
    dst.d=float3fromVector(src.d);
    dst.rxOrigin=float3fromPoint(src.rxOrigin);
    dst.rxDirection=float3fromVector(src.rxDirection);
    dst.ryOrigin=float3fromPoint(src.ryOrigin);
    dst.ryDirection=float3fromVector(src.ryDirection);
}

void PbrtCamera::init(Camera* c, Sampler* s){
    camera=c;
    sampler=s;
    const string progPTX=ptxpath("util", "pbrtcamera.cu");
    progCamera=gContext->createProgramFromPTXFile(progPTX,
        "cameraRay");

    int xstart,xend,ystart,yend;
    camera->film->GetSampleExtent(&xstart, &xend, &ystart, &yend);

    width=xend-xstart;
    height=yend-ystart;
    samplesPerPixel=sampler->samplesPerPixel;
}


PbrtCamera::~PbrtCamera()
{
    progCamera->destroy();
}

void PbrtCamera::getExtent(RTsize& rWidth, RTsize& rHeight)
{
    int sppWidth=samplesPerPixel;
    int sppHeight=1;

    while((sppWidth>=sppHeight) && ((sppWidth&0x1)==0x0)){
        sppWidth/=2; // compiler should smart enough
        sppHeight*=2;
    }

    rWidth=static_cast<RTsize>(width*sppWidth);
    rHeight=static_cast<RTsize>(height*sppHeight);
}

optix::Program PbrtCamera::getRayProg()
{
    return progCamera;
}

void PbrtCamera::preLaunch(const Scene* scene)
{
    //MemoryArena arena;
    RNG rng;
    int maxSamples = sampler->MaximumSampleCount();
    Sample* sample=new Sample(sampler, nullptr, nullptr, scene);
    Sample* samples=sample->Duplicate(maxSamples);

    int maximalSamples=width*height*samplesPerPixel;
    csamples=new CameraSample[maximalSamples];

    optix::Buffer bRays=gContext->createBuffer(RT_BUFFER_INPUT);
    bRays->setFormat(RT_FORMAT_USER);
    bRays->setElementSize(sizeof(CudaRayDifferential));
    bRays->setSize(width,height,samplesPerPixel);

    CudaRayDifferential* pRays=reinterpret_cast<CudaRayDifferential*>(bRays->map());
    int sampleCount, currSample=0;
    while ((sampleCount = sampler->GetMoreSamples(samples, rng)) > 0) {
        RayDifferential rd;
        for (int i=0; i < sampleCount; ++i) {
            if (currSample>=maximalSamples){
                Severe("Too many samples. expect:%d", maximalSamples);
                //to return in last statement
            }
            float rayWeight = camera->GenerateRayDifferential(samples[i], &rd);
            rd.ScaleDifferentials(1.f / sqrtf(samplesPerPixel));
            initCudaRDFromRD(pRays[currSample], rd);
            csamples[currSample]=samples[i];
            currSample++;
        }
    }
    if (currSample != width*height*samplesPerPixel){
        Warning("pbrt camera do not generate enough samples");
    }

    bRays->unmap();
    gContext["bRays"]->set(bRays);
}

void PbrtCamera::postLaunch()
{
    bRay->destroy();
    delete csamples;

}

CameraSample* PbrtCamera::getCameraSamples()
{
    return csamples;
}
