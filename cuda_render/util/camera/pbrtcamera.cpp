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

void PbrtCamera::init(Camera* c, Sampler* s, CudaSample* sa){
    camera=c;
    sampler=s;
    cudaSample=sa;
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

    while((sppWidth>sppHeight) && ((sppWidth&0x1)==0x0)){
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
   
    Sample* samples=cudaSample->sample->Duplicate(maxSamples);

    int maximalSamples=width*height*samplesPerPixel;
    csamples=new CameraSample[maximalSamples];

    bRays=gContext->createBuffer(RT_BUFFER_INPUT);
    bRays->setFormat(RT_FORMAT_USER);
    bRays->setElementSize(sizeof(CudaRayDifferential));

    RTsize bWidth;
    RTsize bHeight;
    getExtent(bWidth, bHeight);
    bRays->setSize(bWidth,bHeight);
    CudaRayDifferential* pRays=reinterpret_cast<CudaRayDifferential*>(bRays->map());

    RTsize sample1Dsize=cudaSample->Sample1DOffset;
    bRandom1D=gContext->createBuffer(RT_BUFFER_INPUT);
    bRandom1D->setFormat(RT_FORMAT_FLOAT);
    bRandom1D->setSize(sample1Dsize, bWidth, bHeight);
    float* pRandom1D=reinterpret_cast<float*>(bRandom1D->map());

    RTsize sample2Dsize=cudaSample->Sample2DOffset;
    bRandom2D=gContext->createBuffer(RT_BUFFER_INPUT);
    bRandom2D->setFormat(RT_FORMAT_FLOAT2);
    bRandom2D->setSize(sample2Dsize, bWidth, bHeight);
    optix::float2* pRandom2D=reinterpret_cast<optix::float2*>(bRandom2D->map());

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
            if(sample1Dsize>0){
                memcpy(pRandom1D, (samples[i].oneD)[0], sample1Dsize*sizeof(float));
                pRandom1D+=sample1Dsize;
            }
            if(sample2Dsize>0){
                memcpy(pRandom2D, (samples[i].twoD)[0], cudaSample->Sample2DOffset*sizeof(optix::float2));
                pRandom2D+=sample2Dsize;
            }
            currSample++;
        }
    }
    if (currSample != width*height*samplesPerPixel){
        Warning("pbrt camera do not generate enough samples");
    }
    bRandom1D->unmap();
    bRandom2D->unmap();
    bRays->unmap();
    gContext["bRays"]->set(bRays);
    gContext["bRandom1D"]->set(bRandom1D);
    gContext["bRandom2D"]->set(bRandom2D);
}

void PbrtCamera::postLaunch()
{
 /*   bRandom1D->destroy();
    bRandom2D->destroy();
    bRays->destroy();
*/
    bRandom1D->setSize(0,0,0);
    bRandom2D->setSize(0,0,0);
    bRays->setSize(0,(RTsize)0);
    //delete csamples;

}

CameraSample* PbrtCamera::getCameraSamples()
{
    return csamples;
}
