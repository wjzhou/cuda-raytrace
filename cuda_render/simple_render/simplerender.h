#ifndef simplerender_h__
#define simplerender_h__

//This is mostly used for test the shape code
#include "cudarender.h"
#include "core/camera.h"
#include "util/camera/cudacamera.h"
class SimpleRenderer : public CudaRenderer
{
public:
    SimpleRenderer(Sampler* sampler, Camera* camera, const ParamSet &params);
    ~SimpleRenderer();
    void render(const Scene* scene, CudaRender* cudarender,
        CudaCamera* cudacamera);
private:
    Sampler *sampler;
    Camera *camera;
};
#endif // simplerender_h__