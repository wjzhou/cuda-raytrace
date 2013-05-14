#include "optix_world.h"

#include "util/light/cudalight.cu.h"
#include "util/util.cu.h"

#include "util/material/cudamaterial.cu.h"
#include "photonmapping.h"
using namespace optix;

rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_group, , );


rtDeclareVariable(uint*, haltonPermute,,);
__device__ static const uint b[]={2,3,5,7,11,13}; 

enum SampleIndex{LU1=0,LU2, U1,U2, NUM_SAMPLEINDEX};

__device__ __inline__ float PermutedRadicalInverse(uint n, uint base, uint* p) 
{
    float val = 0;
    float invBase = 1.f / base, invBi = invBase;

    while (n > 0) {
        uint d_i = p[n % base];
        val += d_i * invBi;
        n *= invBase;
        invBi *= invBase;
    }
    return val;
}

//static const float OneMinusEpsilon=0x1.fffffep-1;
__device__ __inline__ void haltonSample(uint n, float* out)
{
    uint *p = haltonPermute;
    for (uint i = 0; i < NUM_SAMPLEINDEX; ++i) {
        //out[i] = min(PermutedRadicalInverse(n, b[i], p), 
         //   OneMinusEpsilon);
        out[i] = PermutedRadicalInverse(n, b[i], p);
        p += b[i];
    }
}

//use a standard halton sequence (a.k.a non-permuted) version at this moment
//this is determined and generate (1/2, 1/3, 1/5, 1/7) fot the (0,0) 
/*
__device__ __inline__ void haltonSample(uint n, float* out)
{
    for (uint i=0; i<dims; ++i){
        uint n1=n+1;
        float val = 0;
        uint base=b[i];
        float invBase = 1. / base, invBi = invBase;
        while (n1 > 0) {
            val += (n1%base) * invBi;
            n1 *= invBase;
            invBi *= invBase;
        }
        out[i]=val;
    }
}*/

rtDeclareVariable(uint, lightSourceIndex, , );
rtDeclareVariable(uint, photonTracinglaunchWidth, , );
rtDeclareVariable(uint, max_photon_count, ,);



struct PhotonTraingPayLoad{
    CudaSpectrum alpha;
    bool specularPath;
    uint nIntersections;
    uint pm_index;
};

rtDeclareVariable(CudaPhoton*, indirectPhotonmap, , );
rtDeclareVariable(float*, photonTracingRandom, , );

RT_PROGRAM void photontracing_camera()
{
    uint  pm_index = (launchIndex.y * photonTracinglaunchWidth + launchIndex.x) * max_photon_count;
    float sample[NUM_SAMPLEINDEX];
    haltonSample(pm_index, sample);
    //rtPrintf("3");
    optix::Ray photonRay;
    float3 N1;
    float pdf;
    CudaSpectrum Le=Sample_L(lightSourceIndex, sample[LU1], sample[LU2], 
        sample[U1], sample[U2], &photonRay, &N1, &pdf);
    if (pdf==0.0f || isBlack(Le)) return;
    for (uint i=0; i<max_photon_count; ++i){
        setInValid(indirectPhotonmap[pm_index+i]);
    }
    photonRay.ray_type=PM_PhotonTracingType;
    photonRay.tmax=RT_DEFAULT_MAX;
    CudaSpectrum alpha = (AbsDot(N1, photonRay.direction) * Le) / (pdf);

    //rtPrintf("\nclpha: %f %f %f, Le: %f %f %f, dot: %f pdf:%f", 
    //    alpha.x, alpha.y, alpha.z, Le.x, Le.y, Le.z, AbsDot(N1, photonRay.direction), pdf);
    PhotonTraingPayLoad pld={alpha, true, 0, pm_index};
    rtTrace(top_group, photonRay, pld);
}
__inline__ __device__ bool materialHasNonSpecular()
{
    return true;
}

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t, rtIntersectionDistance, );
rtDeclareVariable(PhotonTraingPayLoad, photonTracingPayLoad, rtPayload, );

RT_PROGRAM void photontracing_closest_hit()
{
    float3 hit_point = ray.origin + t*ray.direction;
    //float3 world_shading_normal   = normalize(rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal));

    rtPrintf("\nnIntersections:%d", photonTracingPayLoad.nIntersections);

    if (isSpecular(materialType)){
        Ray newRay;
        CudaSpectrum spec=materialSpecular(-ray.direction, &newRay.direction, false,hit_point);
        newRay.origin=hit_point;
        newRay.ray_type=PM_PhotonTracingType;
        newRay.tmin=scene_epsilon;
        newRay.tmax=RT_DEFAULT_MAX;
        photonTracingPayLoad.alpha*=spec;
        if(photonTracingPayLoad.nIntersections==0){
            photonTracingPayLoad.nIntersections++;
        }
        rtTrace(top_group, newRay, photonTracingPayLoad);
        return;
    }

    float3 wo = -ray.direction;

    //rtPrintf("\n\nwhadingNormal:%f %f %f", world_shading_normal.x, world_shading_normal.y, world_shading_normal.z);

    
    uint nIntersections=photonTracingPayLoad.nIntersections;
    if (materialHasNonSpecular()){
        //bool depositedPhoton = false;
        if (nIntersections >= 1) { //at least bounced once
            CudaPhoton& photon=indirectPhotonmap[photonTracingPayLoad.pm_index+nIntersections-1];
            setValid(photon);
            photon.p=hit_point;
            photon.alpha= photonTracingPayLoad.alpha;
            photon.wi=wo;
            //depositedPhoton = true;
        }
    }
    
    if (nIntersections >= max_photon_count) {
        return;
    }

    float3 wiw;
    float pdf;
    uint ranomIndex=3*(photonTracingPayLoad.pm_index+nIntersections);
    float u1=photonTracingRandom[ranomIndex];
    float u2=photonTracingRandom[ranomIndex+1];
    CudaSpectrum fr = Sample_f(wo, &wiw, u1, u2, &pdf);

    if (isBlack(fr) || pdf == 0.f) return;
    float3 wShadingNormal = normalize(
        rtTransformNormal(RT_OBJECT_TO_WORLD, aShadingNormal));
    CudaSpectrum anew = photonTracingPayLoad.alpha * fr *
        AbsDot(wiw, wShadingNormal) / pdf;

    rtPrintf("\nanew:%f %f %f, %f, %f", anew.x, anew.y, anew.z, AbsDot(wiw, wShadingNormal), pdf);


    // Possibly terminate photon path with Russian roulette
    /*float continueProb = min(1.f, anew.y / photonTracingPayLoad.alpha.y); //the pbrt use the y() value
    if (photonTracingRandom[ranomIndex+2] > continueProb)
        return;
    photonTracingPayLoad.alpha = anew / continueProb;
    */
    photonTracingPayLoad.alpha=anew;
    photonTracingPayLoad.nIntersections++;

    Ray newRay = Ray(hit_point, wiw, PM_PhotonTracingType,
        scene_epsilon);
    rtTrace(top_group, newRay, photonTracingPayLoad);
}

RT_PROGRAM void  photontracing_exception()
{
    rtPrintExceptionDetails();
}

RT_PROGRAM void  photontracing_miss()
{
    rtPrintf("\nmiss");
}

RT_PROGRAM void  photontracing_debug_anyhit()
{
}