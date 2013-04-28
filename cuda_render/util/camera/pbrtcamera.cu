#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "util/common.cu.h"

using namespace optix;
/*
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtBuffer<CudaRayDifferential, 2> bRays;
RT_CALLABLE_PROGRAM CudaRayDifferential cameraRay()
{
    return bRays[launchIndex];
}
*/

RT_CALLABLE_PROGRAM void cameraRay()
{

}

// Stubs only needed for sm_1x
#if __CUDA_ARCH__ < 200
__global__ void checker_color_stub()
{
  cameraRay();
}
#endif