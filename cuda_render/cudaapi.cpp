#include "cudarender.h"

static CudaRender cudaRender;
void CudaRenderInit()
{
    cudaRender.init();
}

void CreateCudaShape(const std::string& name, Reference<Shape>& shape,
    std::vector<Reference<Primitive> >* currentInstance,
    const Material* material)
{
    cudaRender.createCudaShape(name, shape, currentInstance, material);

}

void CudaObjectInstance( std::vector<Reference<Primitive> >* key, const Transform& transform)
{
    cudaRender.objectInstance(key, transform);
}

Renderer* CreateCudaRenderer(Sampler* sampler, Camera* camera, const ParamSet& params )
{
    cudaRender.createSubRenderer(sampler, camera, params, "simple");
    return &cudaRender;
}