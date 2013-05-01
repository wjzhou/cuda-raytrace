#include "cudamaterial.h"
#include "core/pbrt.h"
#include "util/util.h"
#include "cudarender.h"
#include <cuda.h>

CudaMaterial* CudaMaterial::createCudaMeteral(const Material* material)
{
    if (const MatteMaterial* mm=dynamic_cast<const MatteMaterial*>(material)){
        return new CudaMatteMaterial(mm);
    }
    return  new CudaMatteMaterial(nullptr);
}

void CudaMatteMaterial::setupMaterial(optix::GeometryInstance instance)
{
    DifferentialGeometry dg;
    CudaSpectrum kd;
    if(matte!=nullptr){
        kd=CudaSpectrumFromSpectrum(matte->Kd->Evaluate(dg))*INV_PI;
    }else{
        kd=CudaSpectrumFromFloat(0.5f);
    }
    CUdeviceptr parameter;
    
    CUresult  result=cuMemAlloc(&parameter, sizeof(CudaSpectrum));
    if(result != CUDA_SUCCESS){
        Severe("material cuda malloc error:%d", result);
    }

    CUresult  result2=cuMemcpyHtoD(parameter, &kd, sizeof(CudaSpectrum));

    instance["materialParameter"]->setUserData(sizeof(CUdeviceptr), &parameter);
    printf("%ld\n", parameter);
    MaterialType mt=MaterialTypeMatt;
    instance["materialType"]->setUserData(sizeof(MaterialType), &mt);
}

CudaMatteMaterial::CudaMatteMaterial(const MatteMaterial* mm) 
    :matte(mm)
{

}
