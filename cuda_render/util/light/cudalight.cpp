#include "cudalight.h"
#include <lights/point.h>
#include "cudarender.h"
#include "util/util.h"
#include "lights/diffuse.h"
#include "shapes/disk.h"
#include "cuda.h"
#include "../random/cudarandom.h"
#include "../cuda/helper_cuda.h"

template <class T>
void CudaLight::setupLight(T* light, CudaSample* sample){
    Warning("UnImplemented Cuda Light Source:%s", typeid(light).name());
}

template <>
void CudaLight::setupLight<PointLight>(PointLight* pl, CudaSample* sample){
    CudaLightDevice cld;
    cld.lt=CudaLightDevice::POINT;
    cld.o=float3fromPoint(pl->lightPos);
    cld.intensity=CudaSpectrumFromSpectrum(pl->Intensity);
    addLight(cld);
}

template <>
void CudaLight::setupLight<DiffuseAreaLight>(DiffuseAreaLight* al, CudaSample* sample){
    CudaLightDevice cld;
    ShapeSet* set=al->shapeSet;
    auto shapes=set->shapes;
    uint32_t samplePerShape=max(1.0f, ceilf(al->nSamples/shapes.size()));
    for(auto it=shapes.begin(); it!=shapes.end(); ++it){
        if (const Disk* disk=dynamic_cast<const Disk*>((*it).GetPtr())){
            Vector worldx=(*disk->ObjectToWorld)(Vector(disk->radius, 0.f, 0.f));
            Vector worldy=(*disk->ObjectToWorld)(Vector(0.f, disk->radius, 0.f));
            Vector normal=Normalize(Cross(worldx, worldy));
            Point worldo=(*disk->ObjectToWorld)(Point(0.f, 0.f, disk->height));
            /*int start=addAux(make_float4(worldo.x, worldo.y,  worldo.z, normal.x));
            addAux(make_float4(worldx.x, worldx.y,  worldx.z, normal.y));
            addAux(make_float4(worldy.x, worldy.y,  worldy.z, normal.z));*/
            cld.lt=CudaLightDevice::AREA_DISK;
            cld.o=float3fromPoint(worldo);
            cld.p1=float3fromVector(worldx);
            cld.p2=float3fromVector(worldy);
            cld.normal=float3fromVector(normal);
            cld.intensity=CudaSpectrumFromSpectrum(al->Lemit*((*it)->Area()));
            cld.randomStart=sample->Add2D(samplePerShape);
            cld.nSample=samplePerShape;
            addLight(cld);
        }else{
             Warning("UnImplemented Cuda Area Light Source:%s", typeid(*it).name());
        }
    }
    //addLight(cld);
}

template <>
void CudaLight::setupLight<Light>(Light* light, CudaSample* sample){
    if(PointLight* l=dynamic_cast<PointLight*>(light)){
        setupLight(l, sample);
    }else if (DiffuseAreaLight* l=dynamic_cast<DiffuseAreaLight*>(light)){
        setupLight(l, sample);
    }else{
        Warning("UnImplemented Cuda Light Source:%s", typeid(*light).name());
        return;
    }
}

void CudaLight::addLight( const CudaLightDevice& cld )
{
    lights.push_back(cld);
}

int CudaLight::addAux( const optix::float4& data )
{
    int totalAux=lightsAux.size();
    lightsAux.push_back(data);
    return totalAux;
}

void CudaLight::postLaunch()
{
    bLights->destroy();
    bLightsAux->destroy();
    lights.clear();
    lightsAux.clear();
    if(dLightRandom2D!=0){
        checkCudaErrors(cuMemFree(dLightRandom2D));
    }
}

void CudaLight::preLaunch(const Scene* scene, CudaSample* sample, unsigned int width, unsigned int height)
{
    for (auto it=scene->lights.begin(); it!=scene->lights.end(); ++it){
        CudaLight::setupLight(*it, sample);
    }

    bLights=gContext->createBuffer(RT_BUFFER_INPUT);
    bLights->setFormat(RT_FORMAT_USER);
    bLights->setElementSize(sizeof(CudaLightDevice));
    bLights->setSize(lights.size());
    CudaLightDevice* pLights= reinterpret_cast<CudaLightDevice*> (bLights->map());
    for (int i=0; i<lights.size(); ++i){
        pLights[i]=lights[i];
    }
    bLights->unmap();
    gContext["bLights"]->set(bLights);

    bLightsAux=gContext->createBuffer(RT_BUFFER_INPUT);
    bLightsAux->setFormat(RT_FORMAT_FLOAT4);
    bLightsAux->setSize(lightsAux.size());
    if(lightsAux.size() > 0){
        optix::float4* pLightsAux= reinterpret_cast<optix::float4*> (bLightsAux->map());
        for (int i=0; i<lightsAux.size(); ++i){
            pLightsAux[i]=lightsAux[i];
        }
        bLightsAux->unmap();
    }
    gContext["bLightsAux"]->set(bLightsAux);
    
    //because my code do not use the 1Dsample yet, put a assert here if I finnal need it
    assert(sample->Sample1DOffset==0);
    unsigned int samples=(2*sample->Sample2DOffset)*width*height;
    dLightRandom2D=0;
    if(samples>0){
        checkCudaErrors(cuMemAlloc(&dLightRandom2D, samples*sizeof(float)));
        CudaRandom rng(2047);
        rng.generate((float*)dLightRandom2D, samples);
        //actually, I do not need input, but seems I do not have the RT_FORMATE_NONE option
        //Must have RT_BUFFER_INPUT here, otherwise the optix assert will fail
        bLightRandom=gContext->createBufferForCUDA(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, width, height, (sample->Sample2DOffset));
        int currentDevice=0;
        checkCudaErrors(cuCtxGetDevice(&currentDevice));
        bLightRandom->setDevicePointer(currentDevice, dLightRandom2D);
        gContext["bLightRandom2D"]->setBuffer(bLightRandom);        
    }
}

