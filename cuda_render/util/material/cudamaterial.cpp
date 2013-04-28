#include "cudamaterial.h"
#include "core/pbrt.h"
#include "util/util.h"
#include "cudarender.h"

CudaMaterial* CudaMaterial::createCudaMeteral(const Material* material)
{
    if (const MatteMaterial* mm=dynamic_cast<const MatteMaterial*>(material)){
        return new CudaMatteMaterial(mm);
    }
    return nullptr;
}

void CudaMatteMaterial::setupMaterial(optix::GeometryInstance instance)
{
    DifferentialGeometry dg;
    CudaSpectrum kd=CudaSpectrumFromSpectrum(matte->Kd->Evaluate(dg))*INV_PI;
    instance["kd"]->setUserData(sizeof(CudaSpectrum), &kd);
    MaterialType mt=MaterialTypeMatt;
    instance["materialType"]->setUserData(sizeof(MaterialType), &mt);
}

CudaMatteMaterial::CudaMatteMaterial(const MatteMaterial* mm) 
    :matte(mm)
{

}
