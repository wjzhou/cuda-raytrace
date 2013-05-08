#include "optix_world.h"
#include "photonmapping.h"
#include "util/material/cudamaterial.cu.h"
#include "util/util.cu.h"
#include "util/light/cudalight.cu.h"
rtBuffer<RayTracingRecord, 2> bRayTracingOutput;
rtBuffer<float3, 2>  bOutput;
rtDeclareVariable(CudaPhoton*, indirectPhotonmap, , );
#define MAX_DEPTH 20

__device__ __inline__ float kernel(float dist2, float maxDist2)
{
     float s = (1.f - dist2 / maxDist2);
     return 3.f * INV_PI * s * s;
}

__device__ __inline__ CudaSpectrum processPhoton(const RayTracingRecord& record, const CudaPhoton& photon, float dist2, float maxDist2)
{
    float k = kernel(dist2, maxDist2);
    if(isPrint()){
        rtPrintf("\nalpha:%f, %f, %f",photon.alpha.x, photon.alpha.y, photon.alpha.z);
    }
    return k/(maxDist2)*fabs(dot(record.shadingNormal, photon.wi))*f(record.material, record.materialParameter, -record.direction, photon.wi)*photon.alpha;
}

__device__ __inline__ CudaSpectrum KdTreeLookup(RayTracingRecord& rec, int nLookup, float maxDist2)
{
    unsigned int stack[MAX_DEPTH];
    unsigned int stack_current = 0;
    unsigned int nodeNum = 0;
    CudaSpectrum L=black();
    //float3& p=rec.position;
    float3 p=make_float3(0.f,0.f,0.f);
#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]
    push_node( 0 );
    do {
        if(nodeNum>(1<<12)){
            nodeNum=pop_node();
            continue;
        }
        CudaPhoton* node=&indirectPhotonmap[nodeNum];
        int axis=node->splitAxis;

        float dist2=DistanceSquared(node->p, p);
        if(dist2 < maxDist2){
              L+=processPhoton(rec, *node, dist2, maxDist2);
        }
        

        if(axis<3){
            /*float p_axis, nodeP_axis;
            switch (axis)
            {
            case 0:
                p_axis=p.x;
                nodeP_axis=node->p.x;
                break;
            case 1:
                p_axis=p.y;
                nodeP_axis=node->p.y;
                break;
            case 2:
                p_axis=p.z;
                nodeP_axis=node->p.z;
                break;
            }*/
            float p_axis=reinterpret_cast<float*>(&p)[axis];
            float nodeP_axis=reinterpret_cast<float*>(&node->p)[axis];
            float dist2 = (p_axis - nodeP_axis)*(p_axis - nodeP_axis);
            
            if (p_axis <= nodeP_axis){
                if (dist2 < maxDist2 && node->rightChild < PHOTON_MAX_RIGHT_CHILD){
                    push_node(node->rightChild); 
                }
                if (node->hasLeftChild){
                    nodeNum=nodeNum+1;
                }else{
                    nodeNum=pop_node();
                }

            } else {
                if (dist2 < maxDist2 && node->hasLeftChild){
                    push_node(nodeNum+1);
                }
                if (node->rightChild < PHOTON_MAX_RIGHT_CHILD){
                    nodeNum=node->rightChild;
                }else{
                    nodeNum=pop_node();
                }
            }
        }
        else{
            nodeNum=pop_node();
        }

    } while (nodeNum);
    return L;
#undef push_node
#undef pop_node
}

__device__ __inline__ CudaSpectrum LPhoton(RayTracingRecord& rec, int nLookup, float maxDist2)
{
    return KdTreeLookup(rec, nLookup, maxDist2);
    //return black();
}

struct ShadowPRD
{
    float attenuation;
};
rtDeclareVariable(rtObject, top_group, ,);
rtDeclareVariable(float, scene_epsilon, ,);
__device__ __inline__ CudaSpectrum directLight(const RayTracingRecord& rec){
    CudaSpectrum L=black();
    const float3& point=rec.position;

    float3 world_shading_normal = rec.geometryNormal;
    float3 world_geometry_normal = rec.shadingNormal;

    int totalLight=lightSize();
    //rtPrintf("1 %f,%f,%f\n", L.x,L.y,L.z);
    for (int i=0; i<totalLight; ++i)
    {
        float3 uwi;
        float pdf;
        CudaSpectrum li=Sample_L(i, point, uwi, pdf);
        Ray shadowRay(point, uwi, PM_ShadowRayType, 0.001f, 1.0f-0.001f);
        ShadowPRD pld;
        pld.attenuation=1.0f;
        rtTrace(top_group, shadowRay, pld);
        float3 wi=normalize(uwi);
        float3 wo=normalize(-rec.direction);
        L+=pld.attenuation*fabs(dot(world_shading_normal, wi))*f(rec.material, rec.materialParameter, wo, wi)*li;
        //rtPrintf("2 %f,%f,%f,%f\n", L.x,L.y,L.z, li.x);
    }
    return L;
}

RT_PROGRAM void gathering_camera(){
    RayTracingRecord& rec=bRayTracingOutput[launchIndex];

    int rec_flags=rec.flags;
    // Check if this is hit point lies on an emitter or hit background 
    if( (rec_flags & RayTracingRecordFlageMISS) || (rec_flags & RayTracingRecordFlageException )) {
        bOutput[launchIndex] = make_float3(0.0f,0.0f,0.0f);
        return;
    }
    //rtPrintf("%ld", (long long)indirectPhotonmap);
    
    //bOutput[launchIndex]=directLight(rec)+LPhoton(rec, 8, 10.0f);
    CudaSpectrum DL=directLight(rec);
    CudaSpectrum IDL=LPhoton(rec, 8, 50.0f);
    if(isPrint()){
        rtPrintf("\nDL:%f %f %f, IDL: %f %f %f", DL.x, DL.y, DL.z, IDL.x, IDL.y, IDL.z);
    }
    //bOutput[launchIndex]=DL+IDL;
    bOutput[launchIndex]=DL+IDL;
}