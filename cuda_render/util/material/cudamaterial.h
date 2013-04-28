#ifndef cudamaterial_h__
#define cudamaterial_h__
#include "optix_world.h"
#include "core/material.h"
#include "materials/matte.h"
#include "../common.cu.h"
#include "core/texture.h"
#include "diffgeom.h"
#include "../util.h"
class CudaMaterial{
public:
    virtual void setupMaterial(optix::GeometryInstance instance)=0;
    virtual ~CudaMaterial(){}
    static CudaMaterial* createCudaMeteral(const Material* material);
};
//The Cuda Material is more like the concept of surface integrator in pbrt
//The pbrt material is transfer to optix::Program progF, which coresponding
//to Matrial::f() in pbrt

class CudaMatteMaterial : public CudaMaterial{
public:
    CudaMatteMaterial(const MatteMaterial* mm);
    virtual void setupMaterial(optix::GeometryInstance instance);
private:
    const MatteMaterial* matte;

    //static optix::Program progF;
    //static void init();

};
#endif // cudamaterial_h__