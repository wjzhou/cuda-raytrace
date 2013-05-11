#ifndef cudamaterial_h__
#define cudamaterial_h__
#include "optix_world.h"
#include "core/material.h"
#include "materials/matte.h"
#include "../common.cu.h"
#include "core/texture.h"
#include "diffgeom.h"
#include "../util.h"
#include "materials/mirror.h"
#include "materials/glass.h"
class CudaMaterial{
public:
    virtual void setupMaterial(optix::GeometryInstance instance)=0;
    virtual ~CudaMaterial(){}
    static CudaMaterial* createCudaMeteral(const Material* material);
};
//The optix Material is more like the concept of surface integrator in pbrt
//The pbrt material is transform to an inline function
//The material needs to set the materialType and materialParameter in the
//optix instance

class CudaMatteMaterial : public CudaMaterial{
public:
    CudaMatteMaterial(const MatteMaterial* mm);
    virtual void setupMaterial(optix::GeometryInstance instance);
    static CudaMatteMaterial defaultMaterial;
private:
    const MatteMaterial* matte;//< this maybe null, if object is 
                               //< used as fall back material
    CUdeviceptr parameter;
    //static optix::Program progF;
    //static void init();
};

class CudaMirrorMaterial : public CudaMaterial{
public:
    CudaMirrorMaterial(const MirrorMaterial* mm);
    virtual void setupMaterial(optix::GeometryInstance instance);
private:
    CUdeviceptr parameter;
};

class CudaGlassMaterial : public CudaMaterial{
public:
    CudaGlassMaterial(const GlassMaterial* mm);
    virtual void setupMaterial(optix::GeometryInstance instance);
private:
    CUdeviceptr parameter;
};

#endif // cudamaterial_h__