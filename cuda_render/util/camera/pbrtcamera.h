#ifndef pbrtcamera_h__
#define pbrtcamera_h__

#include "sampler.h"
#include "camera.h"
#include "camera.cu.h"
#include "cudacamera.h"
void initCudaRDFromRD(CudaRayDifferential& dst, const RayDifferential& src);

    /* adaptive sampler is not supported, and I think the adaptive sampler
     * itself has a bug, that the normal pixel(not super sampled one) has
     * a wrong scale in the ray[i].ScaleDifferentials
     */
class PbrtCamera : public CudaCamera{
public:
    ~PbrtCamera();
    virtual void init(Camera* camera, Sampler* sampler);
    void getExtent(RTsize& rWidth, RTsize& rHeight);
    CameraSample* getCameraSamples();
    optix::Program getRayProg();
    void preLaunch(const Scene* scene);
    void postLaunch();
private:
    Sampler* sampler;
    Camera* camera;
    CameraSample* csamples;
    optix::Buffer bRays;
    optix::Program progCamera;
    int width;
    int height;
    int samplesPerPixel;
};
#endif // pbrtcamera_h__