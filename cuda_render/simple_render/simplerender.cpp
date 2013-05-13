#include "simplerender.h"
#include "util/util.h"
#include "util/camera/cudacamera.h"
#include "core/camera.h"
#include "core/film.h"
#include "util/light/cudalight.h"
SimpleRenderer::SimpleRenderer(Sampler* s, Camera* c, const ParamSet& params)
    :sampler(s), camera(c)
{

}

SimpleRenderer::~SimpleRenderer()
{

}

void SimpleRenderer::render(const Scene* scene, CudaRender* cudarender,
    CudaCamera* cudacamera)
{

    gContext->setStackSize(1024);
    gContext->setRayTypeCount(2);
    gContext->setEntryPointCount(1);
    gContext["scene_epsilon"]->setFloat(0.01f);
    const string progPTX=ptxpath("simple_render", "simplerender.cu");
    gContext->setMissProgram(0, gContext->createProgramFromPTXFile(progPTX,
        "simple_miss"));

    optix::Program progClosestHit=gContext->createProgramFromPTXFile(progPTX,
        "simple_cloest_hit");
    optix::Program progShadowAnyHit=gContext->createProgramFromPTXFile(progPTX,
        "simple_shadow_any_hit");

    for (std::vector<optix::GeometryInstance>::iterator
        it=cudarender->geometryInstances.begin();
        it!= cudarender->geometryInstances.end(); ++it)
    {
        (*it)->setMaterialCount(1);
        optix::Material m=gContext->createMaterial();
        m->setClosestHitProgram(0, progClosestHit);
        m->setAnyHitProgram(1, progShadowAnyHit);
        (*it)->setMaterial(0, m);
    }

    //Sample* sample=new Sample(sampler, nullptr, nullptr, scene);
    CudaSample* cudaSample=new CudaSample(sampler);
    
    cudacamera->init(camera, sampler, cudaSample);
    RTsize width;
    RTsize height;
    cudacamera->getExtent(width, height);
    CudaLight light;
    light.preLaunch(scene, cudaSample, width, height);

    optix::Program progCamera=gContext->createProgramFromPTXFile(progPTX,
        "simple_camera");
    //progCamera["cameraRay"]->set(cudacamera->getRayProg());
    gContext->setRayGenerationProgram(0, progCamera);
    cudacamera->preLaunch(scene);

    CameraSample* csamples=cudacamera->getCameraSamples();
    optix::Buffer bOutput=gContext->createBuffer(RT_BUFFER_OUTPUT);
    bOutput->setFormat(RT_FORMAT_FLOAT3);
    bOutput->setSize(width, height);
    gContext["bOutput"]->set(bOutput);

    gContext->launch( 0,
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height)
        );

    optix::float3* pOutput= reinterpret_cast<optix::float3*> (bOutput->map());
    int currSample=width*height;
    //FILE *file = fopen("sm.txt","w");
    for(int i=0; i<currSample; ++i){
        RGBSpectrum result=RGBSpectrum::FromRGB(reinterpret_cast<float*>(&pOutput[i]));
        // Issue warning if unexpected radiance value returned
        if (result.HasNaNs()) {
            Error("Not-a-number radiance value returned "
                "for image sample.  Setting to black.");
            result = Spectrum(0.f);
        }
        else if (result.y() < -1e-5) {
            Error("Negative luminance value, %f, returned"
                "for image sample.  Setting to black.", result.y());
            result=Spectrum(0.f);
        }
        else if (isinf(result.y())) {
            Error("Infinite luminance value returned"
                "for image sample.  Setting to black.");
            result = Spectrum(0.f);
        }
        camera->film->AddSample(csamples[i], result);
        //fprintf(file, "\niX:%f\tiY%f\t", csamples[i].imageX, csamples[i].imageY);
        //result.Print(file);
    }
    //fclose(file);
    light.postLaunch();
    cudacamera->postLaunch();

    camera->film->WriteImage();
}
