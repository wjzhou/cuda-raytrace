#include "util.h"
#include "config.h"
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

