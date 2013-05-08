#ifndef cudashape_cu_h__
#define cudashape_cu_h__

#include <optix_world.h>
using optix::float3;
rtDeclareVariable(float3, geometry_normal, attribute geometry_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, uv, attribute uv, );//tex coordinate
rtDeclareVariable(float3, dpdu, attribute dpdu, );
rtDeclareVariable(float3, dpdv, attribute dpdv, );

#endif // cudashape.cu_h__
