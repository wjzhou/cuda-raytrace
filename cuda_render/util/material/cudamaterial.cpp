#include "cudamaterial.h"
#include "core/pbrt.h"
#include "util/util.h"
#include "cudarender.h"
#include <cuda.h>
#include "../cuda/helper_cuda.h"

CudaMaterial* CudaMaterial::createCudaMeteral(const Material* material)
{
    if (const MatteMaterial* mm=dynamic_cast<const MatteMaterial*>(material)){
        return new CudaMatteMaterial(mm);
    }
    if (const MirrorMaterial* mm=dynamic_cast<const MirrorMaterial*>(material)){
        return new CudaMirrorMaterial(mm);
    }
    if (const GlassMaterial* mm=dynamic_cast<const GlassMaterial*>(material)){
        return new CudaGlassMaterial(mm);
    }

    return  new CudaMatteMaterial(nullptr);
}

void CudaMatteMaterial::setupMaterial(optix::GeometryInstance instance)
{
    instance["materialParameter"]->setUserData(sizeof(CUdeviceptr), &parameter);
    printf("%lld\n", parameter);
    MaterialType mt=MaterialTypeMatt;
    instance["materialType"]->setUserData(sizeof(MaterialType), &mt);
}

CudaMatteMaterial::CudaMatteMaterial(const MatteMaterial* mm) 
    :matte(mm)
{
    checkCudaErrors(cuMemAlloc(&parameter, sizeof(CudaSpectrum)));
    DifferentialGeometry dg;
    CudaSpectrum kd;
    if(matte!=nullptr){
        kd=CudaSpectrumFromSpectrum(matte->Kd->Evaluate(dg));
    }else{
        kd=CudaSpectrumFromFloat(0.5f);
    }
    checkCudaErrors(cuMemcpyHtoD(parameter, &kd, sizeof(CudaSpectrum)));
}

void CudaMirrorMaterial::setupMaterial(optix::GeometryInstance instance)
{
    instance["materialParameter"]->setUserData(sizeof(CUdeviceptr), &parameter);
    //printf("%ld\n", parameter);
    MaterialType mt=MaterialTypeMirror;
    instance["materialType"]->setUserData(sizeof(MaterialType), &mt);
}

CudaMirrorMaterial::CudaMirrorMaterial(const MirrorMaterial* mm) 
{
    checkCudaErrors(cuMemAlloc(&parameter, sizeof(CudaSpectrum)));
    DifferentialGeometry dg;
    CudaSpectrum kr=CudaSpectrumFromSpectrum(mm->Kr->Evaluate(dg));
    checkCudaErrors(cuMemcpyHtoD(parameter, &kr, sizeof(CudaSpectrum)));
}

void CudaGlassMaterial::setupMaterial(optix::GeometryInstance instance)
{
    //instance["materialParameter"]->setUserData(sizeof(CUdeviceptr), &parameter);
    //printf("%ld\n", parameter);
    MaterialType mt=MaterialTypeGlass;
    instance["materialType"]->setUserData(sizeof(MaterialType), &mt);
}

CudaGlassMaterial::CudaGlassMaterial(const GlassMaterial* mm) 
{
    //checkCudaErrors(cuMemAlloc(&parameter, sizeof(CudaSpectrum)));
    DifferentialGeometry dg;
    //CudaSpectrum kr=CudaSpectrumFromSpectrum(mm->Kr->Evaluate(dg));
    //checkCudaErrors(cuMemcpyHtoD(parameter, &kr, sizeof(CudaSpectrum)));
}
