#include "util.h"
#include "config.h"
#include "spectrum.h"
std::string
    ptxpath( const std::string& target, const std::string& base )
{
    static std::string path(CUDA_RENDER_PTX_DIR);
    return path + "/" + target + "_generated_" + base + ".ptx";
}

optix::float3 float3fromNormal(const Normal& n)
{
    optix::float3 ret;
    ret.x=n.x;
    ret.y=n.y;
    ret.z=n.z;
    return ret;
}

optix::float3 float3fromVector(const Vector& v)
{
    optix::float3 ret;
    ret.x=v.x;
    ret.y=v.y;
    ret.z=v.z;
    return ret;
}

optix::float3 float3fromPoint(const Point& p)
{
    optix::float3 ret;
    ret.x=p.x;
    ret.y=p.y;
    ret.z=p.z;
    return ret;
}

Normal normalfromFloat3(const optix::float3& n)
{
    Normal ret;
    ret.x=n.x;
    ret.y=n.y;
    ret.z=n.z;
    return ret;
}

Vector vectorfromFloat3(const optix::float3& v)
{
    Vector ret;
    ret.x=v.x;
    ret.y=v.y;
    ret.z=v.z;
    return ret;
}

Point pointfromFloat3(const optix::float3& p)
{
    Point ret;
    ret.x=p.x;
    ret.y=p.y;
    ret.z=p.z;
    return ret;
}

#define CHECK_TYPE_COMPAT(T, U)                     \
    typedef T TC##__LINE__; typedef U TC##__LINE__

CudaSpectrum CudaSpectrumFromSpectrum(const Spectrum& spec){
     CHECK_TYPE_COMPAT(optix::float3, CudaSpectrum);
     CudaSpectrum result;
     spec.ToRGB(&result.x);
     return result;
}