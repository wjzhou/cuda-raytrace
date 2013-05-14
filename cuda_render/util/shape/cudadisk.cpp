
#include "cudadisk.h"
#include "cudarender.h"
#include "util/util.h"

optix::Program CudaDisk::progBoundingBox;
optix::Program CudaDisk::progIntersection;

CudaDisk::CudaDisk(const Reference<Shape>& shape)
    :shape(shape)
{

}

optix::Geometry CudaDisk::setupGeometry()
{
    const Disk* disk=dynamic_cast<const Disk*>(shape.GetPtr());
    optix::Geometry geomtry=gContext->createGeometry();
    geomtry->setPrimitiveCount(1);
    geomtry->setIntersectionProgram(progIntersection);
    geomtry->setBoundingBoxProgram(progBoundingBox);

    //geomtry["height"]->setFloat(height);
    const Transform& o2w=*disk->ObjectToWorld;
    Point worldo=o2w(Point(0.f,0.f, disk->height));
    Vector worldx=o2w(Vector(disk->radius, 0.f, 0.f));
    Vector worldy=o2w(Vector(0.f, disk->radius, 0.f));
    Vector worldz=o2w(Vector(0.f, 0.f, 1.f));
    worldz=Normalize(worldz);
    float normalizedInnerRadius=disk->innerRadius/disk->radius;

    optix::float2 invRadius2=optix::make_float2(1.f/worldx.LengthSquared(), 1.f/worldy.LengthSquared());
    geomtry["invRadius2"]->set2fv(&invRadius2.x);
    geomtry["innerRadius"]->setFloat(normalizedInnerRadius);
    geomtry["phiMax"]->setFloat(disk->phiMax);

    geomtry["worldx"]->set3fv(&worldx[0]);
    geomtry["worldy"]->set3fv(&worldy[0]);
    geomtry["worldz"]->set3fv(&worldz[0]);
    geomtry["worldo"]->set3fv(&worldo[0]);

    float moffset=Dot(Vector(worldo), worldz);
    geomtry["moffset"]->setFloat(moffset);
    return geomtry;
}

//CUDASHAPE_INIT(disk, CudaDisk);
void CudaDisk::init()
{
    const string diskPTX=ptxpath("util", "cudadisk.cu");
    progBoundingBox = gContext->createProgramFromPTXFile(diskPTX, "disk_bounds" );
    progIntersection = gContext->createProgramFromPTXFile(diskPTX, "disk_intersect" );
}
