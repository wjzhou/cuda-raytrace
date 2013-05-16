#include "optix_world.h"
#include "photonmapping.h"
#include "util/material/cudamaterial.cu.h"
#include "util/util.cu.h"
#include "util/light/cudalight.cu.h"
rtBuffer<RayTracingRecord, 2> bRayTracingOutput;
rtBuffer<float3, 2>  bOutput;
rtDeclareVariable(CudaPhoton*, indirectPhotonmap, , );
#define MAX_DEPTH 40

__device__ __inline__ float kernel(float dist2, float maxDist2)
{
     float s = (1.f - dist2 / maxDist2);
     return 3.f * INV_PI * s * s;
}

__device__ __inline__ CudaSpectrum processPhoton(const RayTracingRecord& record, const CudaPhoton& photon, float dist2, float maxDist2)
{
    float k = kernel(dist2, maxDist2);
    rtPrintf("\nalpha:%f, %f, %f",photon.alpha.x, photon.alpha.y, photon.alpha.z);
    //return k/(maxDist2)*fabs(dot(record.shadingNormal, photon.wi))*f(record.material, record.materialParameter, -record.direction, photon.wi)*photon.alpha;
    return fabs(dot(record.shadingNormal, photon.wi))*f(record.material, record.materialParameter, -record.direction, photon.wi)*photon.alpha;
}

__device__ __inline__ CudaSpectrum KdTreeLookup(RayTracingRecord& rec, int& nLookup, float maxDist2)
{
    unsigned int stack[MAX_DEPTH];
    unsigned int stack_current = 0;
    unsigned int nodeNum = 0;
    CudaSpectrum L=black();
    float3& p=rec.position;
#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]
    push_node( 0 );
    do {
        CudaPhoton* node=&indirectPhotonmap[nodeNum];
        int axis=node->splitAxis;

        float dist2=DistanceSquared(node->p, p);
        if(dist2 < maxDist2){
            nLookup++;
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

__device__ __inline__ CudaSpectrum LPhoton(RayTracingRecord& rec, int& nLookup, float maxDist2)
{
    return KdTreeLookup(rec, nLookup, maxDist2);
    //return black();
}

RT_PROGRAM void photonGatheringCamera(){
    RayTracingRecord& rec=bRayTracingOutput[launchIndex];

    int rec_flags=rec.flags;
    // Check if this is hit point lies on an emitter or hit background 
    if( (rec_flags & RayTracingRecordFlageMISS) || (rec_flags & RayTracingRecordFlageException )) {
        return;
    }
    
    int currentPhotons=0;
    CudaSpectrum IDL=LPhoton(rec, currentPhotons, rec.radius2);
    float alpha=0.7f;
    if(currentPhotons>0){
        int totalPhotons=rec.photon_count+alpha*currentPhotons;
        float ratio=totalPhotons/(rec.photon_count+currentPhotons);
        rec.radius2=rec.radius2*ratio;

        rec.flux=(rec.flux+IDL)*ratio;
        rec.photon_count=totalPhotons;
        rtPrintf("\nradius2:%f, currphotons:%d,  photons:%d IDL: %f %f %f", rec.radius2, currentPhotons, 
            rec.photon_count, rec.flux.x/rec.photon_count, rec.flux.y/rec.photon_count, rec.flux.z/rec.photon_count);
    }
}

rtDeclareVariable(float, emittingPhotons, ,);
RT_PROGRAM void finalGatheringCamera(){

    
    RayTracingRecord& rec=bRayTracingOutput[launchIndex];
    if( (rec.flags & RayTracingRecordFlageMISS) || (rec.flags & RayTracingRecordFlageException )) {
        bOutput[launchIndex] = make_float3(0.0f,0.0f,0.0f);
        return;
    }
    CudaSpectrum DL=rec.directLight;
    CudaSpectrum IDL=black();
    if(rec.photon_count!=0){
        IDL=rec.flux*INV_PI/(rec.radius2*emittingPhotons);
    }
        
    rtPrintf("\n------DL:%f %f %f,emittingPhotons:%f,  IDL: %f %f %f\n-----\n", 
        DL.x, DL.y, DL.z, emittingPhotons, IDL.x, IDL.y, IDL.z);
    bOutput[launchIndex]=DL+IDL;
}