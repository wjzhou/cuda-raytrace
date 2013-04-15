#ifndef util_h__
#define util_h__
#include <string>
#include "core/geometry.h"
#include "optixpp_namespace.h"
std::string
ptxpath( const std::string& target, const std::string& base );

optix::float3 float3fromPoint(const Point& p);
optix::float3 float3fromVector(const Vector& v);
optix::float3 float3fromNormal(const Normal& n);

#endif // util_h__