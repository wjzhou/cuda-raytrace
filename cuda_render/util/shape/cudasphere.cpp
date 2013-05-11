#include "cudasphere.h"
#include <pbrt.h>
#include "cudarender.h"
#include "util/util.h"
#include "optixu_math_namespace.h"

optix::Program CudaSphere::progBoundingBox;
optix::Program CudaSphere::progIntersection;

CudaSphere::CudaSphere(Sphere* sp)
    :sphere(sp)
{
}

optix::Transform CudaSphere::setupTransform()
{
    optix::Geometry geomtry=gContext->createGeometry();
    geomtry->setPrimitiveCount(1);
    geomtry->setIntersectionProgram(progIntersection);
    geomtry->setBoundingBoxProgram(progBoundingBox);
    geomtry["radius"]->setFloat(sphere->radius);
    
    optix::GeometryInstance instance=gContext->createGeometryInstance();
    instance->setGeometry(geomtry);

    optix::Transform tr=gContext->createTransform();
    tr->setMatrix(false, (float*)sphere->ObjectToWorld->GetMatrix().m, 
        (float*)sphere->WorldToObject->GetMatrix().m);
   
    optix::Acceleration acc=gContext->createAcceleration("NoAccel","NoAccel");
    
    optix::GeometryGroup group=gContext->createGeometryGroup();
    group->setChildCount(1);
    group->setChild(0, instance);
    group->setAcceleration(acc);
    tr->setChild(group);

    return tr;
}

void
CudaSphere::init()
{
    const string triangleMeshPTX=ptxpath("util", "cudasphere.cu");
    progBoundingBox = gContext->createProgramFromPTXFile(triangleMeshPTX, "sphere_bounds");
    progIntersection = gContext->createProgramFromPTXFile(triangleMeshPTX, "sphere_intersect");
}
