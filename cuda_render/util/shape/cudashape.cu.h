#ifndef cudashape_cu_h__
#define cudashape_cu_h__

#include <optix_world.h>
using optix::float3;
// Add prefix a(attrubutes) to prevent miss use in function
rtDeclareVariable(float3, aGeometryNormal, attribute aGeometryNormal, );
rtDeclareVariable(float3, aShadingNormal, attribute aShadingNormal, );
rtDeclareVariable(float2, aUv, attribute uv, );//tex coordinate
rtDeclareVariable(float3, aDpdu, attribute aDpdu, );
rtDeclareVariable(float3, aDpdv, attribute aDpdv, );

#endif // cudashape.cu_h__
