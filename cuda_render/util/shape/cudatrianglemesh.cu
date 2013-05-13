#include <optix_world.h>
#include "cudashape.cu.h"
using namespace optix;

rtBuffer<float3> bVertices;
rtBuffer<float3> bNormals;
rtBuffer<float2> bUvs;
rtBuffer<int3>   bIndices;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void trianglemesh_intersect( int primIdx )
{
    int3 idx = bIndices[primIdx];
    float3 p0 = bVertices[idx.x];
    float3 p1 = bVertices[idx.y];
    float3 p2 = bVertices[idx.z];

    float3 n;
    float  t, beta, gamma;
    //use the ray-triangle intersection method from the optix library.
    //The comments of that function memtioned branch-less, thus, it
    //should be more efficient.
    if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)){
        if(rtPotentialIntersection(t)) {
            float2 uv0, uv1, uv2;
            if (bUvs.size() == 0) {
                //assume t0=(0,0), t1=(1,0), t2=(0,1)
                //texcoord = make_float2(beta+gamma);
                uv0=make_float2(0.0f, 0.0f);
                uv1=make_float2(1.0f, 0.0f);
                uv2=make_float2(0.0f, 1.0f);
            } else {
                uv0=bUvs[idx.x];
                uv1=bUvs[idx.y];
                uv2=bUvs[idx.z];
                //texcoord = t1*beta + t2*gamma + t0*(1.0f-beta-gamma) ;
            }

            float du1=uv0.x-uv2.x;
            float du2=uv1.x-uv2.x;
            float dv1=uv0.y-uv2.y;
            float dv2=uv1.y-uv2.y;

            float3 dp1=p0-p2;
            float3 dp2=p1-p2;
            float determinant=du1*dv2-dv1*du2;
            //degenerate triangle, choose the dpdu arbitary
            if (determinant == 0.0f){
                if (fabsf(n.x) > fabsf(n.y)) {
                    float invLen = 1.f/sqrtf(n.x*n.x+n.z*n.z);
                    aDpdu = make_float3(-n.z*invLen, 0.f, n.x*invLen);
                }
                else {
                    float invLen = 1.f/sqrtf(n.y*n.y+n.z*n.z);
                    aDpdu = make_float3(0.f, n.z*invLen, n.y*invLen);
                }
                aDpdv =normalize(cross(n, aDpdu));
            }else{
                float invdet = 1.f / determinant;
                aDpdu = ( dv2 * dp1 - dv1 * dp2) * invdet;
                aDpdv = (-du2 * dp1 + du1 * dp2) * invdet;

            }
             aUv=(1-beta-gamma)*uv0+beta*uv1+gamma*uv2;
             //todo alphaTexture
             aGeometryNormal=n;
             if (bNormals.size() == 0) {
                 aShadingNormal = aGeometryNormal;
             } else {
                 float3 n0 = bNormals[idx.x];
                 float3 n1 = bNormals[idx.y];
                 float3 n2 = bNormals[idx.z];
                 aShadingNormal = n1*beta + n2*gamma + n0*(1.0f-beta-gamma);
                 //shading override
             }
             //p=ray.origin+t*ray.direction;
             rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void trianglemesh_bounds (int primIdx, float result[6])
{
  const int3 idx = bIndices[primIdx];

  const float3 p0 = bVertices[idx.x];
  const float3 p1 = bVertices[idx.y];
  const float3 p2 = bVertices[idx.z];
  const float area = length(cross(p1-p0, p2-p0));

  optix::Aabb* aabb = (optix::Aabb*)result;
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf(fminf(p0, p1), p2);
    aabb->m_max = fmaxf(fmaxf(p0, p1), p2);
  } else {
    aabb->invalidate();
  }
}

