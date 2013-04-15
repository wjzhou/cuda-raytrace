//still not have a good idea about camera, do it later

#ifndef cudacamera_h__
#define cudacamera_h__

#include <optixpp_namespace.h>
#include "core/sampler.h"
class CudaCamera{
public:
    virtual ~CudaCamera(){};
    virtual void init(Camera* camera, Sampler* sampler)=0;
    virtual void getExtent(RTsize& rWidth, RTsize& rHeight)=0;
    virtual optix::Program getRayProg()=0;
    virtual void preLaunch(const Scene* scene)=0;
    virtual void postLaunch()=0;
    virtual CameraSample* getCameraSamples()=0;;
};
#endif // cudacamera_h