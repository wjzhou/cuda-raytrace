#ifndef util_h__
#define util_h__
#include <string>
#include "core/geometry.h"
#include "optixpp_namespace.h"
#include "common.cu.h"
std::string
ptxpath( const std::string& target, const std::string& base );

optix::float3 float3fromPoint(const Point& p);
optix::float3 float3fromVector(const Vector& v);
optix::float3 float3fromNormal(const Normal& n);

Normal normalfromFloat3(const optix::float3& n);
Vector vectorfromFloat3(const optix::float3& v);
Point pointfromFloat3(const optix::float3& p);

CudaSpectrum CudaSpectrumFromSpectrum(const Spectrum& spec);
#endif // util_h__