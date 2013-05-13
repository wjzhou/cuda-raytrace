#ifndef CudaLight_h__
#define CudaLight_h__

#include "../common.cu.h"
#include "optixpp_namespace.h"
#include "scene.h"
#include "../sampler/cudasample.h"

class CudaLight{
public:
    void preLaunch(const Scene* scene, CudaSample* sample, unsigned int width, unsigned int height);
    void postLaunch();

private:
    template <class T> void setupLight(T* light, CudaSample* sample);
    int addAux(const optix::float4& data);
    void addLight(const CudaLightDevice& cld);
    vector<CudaLightDevice> lights;
    vector<optix::float4> lightsAux;
    optix::Buffer bLights;
    optix::Buffer bLightsAux;
    CUdeviceptr dLightRandom2D;
    optix::Buffer bLightRandom;
};

/*class CudaLight{
    virtual void setupLight();
    int addPrivateDate(optix::float4);
    createCudaLight();
};

class CudaPointLight : public CudaLight{
    CudaLightDevice setupLight(){
        CudaLightDevice cld;
        cld.LightType=CudaLightDevice::LightType::POINT;
        cld.d=pl->lightPos;
        cld.intensity=pl->Intensity;
        return cld;
    }
private:
    PointLight* pl;
};
class CudaDiffuseAreaLight : public CudaLight{
    void setupLight();
};*/
#endif // CudaLight_h__
