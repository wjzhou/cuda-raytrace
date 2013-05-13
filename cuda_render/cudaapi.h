/* This file is used to introduce the cudarender into pbrt. It can not contain
 * any cuda head. Because cudarender.h needs classes defined in pbrt, it will
 * introduce recursive dependence if these include is also inside the head
 */
#ifndef cudaapi_h__
#define cudaapi_h__

Renderer* CreateCudaRenderer(Sampler *sampler, Camera *camera,
                             const ParamSet &params,
                             const std::string& rendername);

void CudaRenderInit();

void CreateCudaShape(const std::string &name, Reference<Shape>& shape,
    std::vector<Reference<Primitive> >* currentInstance,
    const Material* material, int lightIndex);

void CudaObjectInstance(std::vector<Reference<Primitive> >* key,
    const Transform& transform);
#endif // cudaapi_h__
