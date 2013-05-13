#ifndef cudarender_h__
#define cudarender_h__
#include "optixpp_namespace.h"
#include <vector>
#include <map>
#include "core/pbrt.h"
#include "core/renderer.h"
#include "core/memory.h"
#include "core/spectrum.h"
#include "util/material/cudamaterial.h"
class CudaCamera;
class CudaRender;

class CudaRenderer {
public:
    virtual void render(const Scene* scene, CudaRender* cudarender,
        CudaCamera* cudacamera)=0;
    virtual ~CudaRenderer(){};
};

extern optix::Context gContext;
class CudaRender : public Renderer{
public:
    CudaRender();
    ~CudaRender(){};
    virtual void Render(const Scene *scene);
    virtual Spectrum Li(const Scene *scene, const RayDifferential &ray,
        const Sample *sample, RNG &rng, MemoryArena &arena,
        Intersection *isect = NULL, Spectrum *T = NULL) const{return Spectrum(0.f);}
    virtual Spectrum Transmittance(const Scene *scene,
        const RayDifferential &ray, const Sample *sample,
        RNG &rng, MemoryArena &arena) const {return Spectrum(0.f);}

    void objectInstance(vector<Reference<Primitive> >* instance,
        const Transform& tr);
    void createCudaShape(const std::string& name, Reference<Shape>& shape,
        vector<Reference<Primitive> >* currentInstance,
        const Material* kMaterial, int lightIndex);
    optix::Group topGroup;
    void createSubRenderer(Sampler *sampler, Camera *camera,
        const ParamSet &params, const std::string& rendername);
    ///< all the geometry will go into this vector(to give subrender chance to set
    ///< optix material(closest hit program)
    std::vector<optix::GeometryInstance> geometryInstances;

private:

    // Do special process for triangles, because they are the major geometries and
    // and unlike other shape, triangles can be intersected in world space easily.
    // Generalize to support all the shapes that use global_space intersection.
    std::vector<optix::GeometryInstance> topGeometryInstances;
    // For other shapes that will do the intersection in object_space
    // and instance of pbrt object instance
    std::vector<optix::Transform> topTransforms;

    // The word Instance has different meaning in pbrt and optix
    // For pbrt, instance is used in object instance. (define a set of objects
    // and reuse the set by object instance). While in optix, instance is simple
    // a combine of material and geometry

    // The key is the RenderOptions.currentInstance
    // convert to void* to avoid strange problem of std::map
    std::map<std::vector<Reference<Primitive> >*, optix::Group> instanceGroups;
    // std::map<void*, optix::Group> instanceGroups;

    // The first instance is for pbrt instance and the second one is optix instance
    std::vector<optix::GeometryInstance> instanceGeometryInstances;
    // similar to the topTransforms but in instance level
    std::vector<optix::Transform> instanceTransforms;

    ///< this is used to detect object instance in pbrt
    vector<Reference<Primitive> >* lastInstance;

    ///< this map store the mapping between pbrt material and cudamaterial
    ///< after render, this should be erase, not implement yet...
    std::map<const Material*, CudaMaterial*> materials;

    CudaRenderer* renderer;
    /// <summary>
    /// This function is called when the current instance is different or right
    /// before the assemble top level group
    /// </summary>
    ///
    optix::Group CudaRender::assembleNode(
        std::vector<optix::GeometryInstance>& geometryInstances,
        std::vector<optix::Transform>& transforms,
        int nReserveChildn=0);
    void assembleObject(std::vector<Reference<Primitive> >* key);
    ///< any sub modular needs init should register here
    void init();
};



#endif // cudarender_h__
