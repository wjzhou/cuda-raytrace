#include "simplerender.h"
#include "util/util.h"
#include "util/camera/cudacamera.h"
#include "core/camera.h"
#include "core/film.h"
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
    gContext->setRayTypeCount(1);
    gContext->setEntryPointCount(1);
    const string progPTX=ptxpath("simple_render", "simplerender.cu");
    gContext->setMissProgram(0, gContext->createProgramFromPTXFile(progPTX,
        "simple_miss"));

    optix::Program progClosestHit=gContext->createProgramFromPTXFile(progPTX,
        "simple_cloest_hit");
    for (std::vector<optix::GeometryInstance>::iterator
        it=cudarender->geometryInstances.begin();
        it!= cudarender->geometryInstances.end(); ++it)
    {
        (*it)->setMaterialCount(1);
        optix::Material m=gContext->createMaterial();
        m->setClosestHitProgram(0, progClosestHit);
        (*it)->setMaterial(0, m);
    }

    cudacamera->init(camera, sampler);
    RTsize width;
    RTsize height;
    cudacamera->getExtent(width, height);

    //optix::Program progCamera=gContext->createProgramFromPTXFile(progPTX,
    //    "simple_camera");
    //progCamera["cameraRay"]->set(cudacamera->getRayProg());
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
    }
    cudacamera->postLaunch();

    camera->film->WriteImage();
}
