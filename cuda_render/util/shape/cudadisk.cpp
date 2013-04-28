
#include "cudadisk.h"
#include "cudarender.h"
#include "util/util.h"
CudaDisk::CudaDisk(const Disk& disk)
    :height(disk.height), radius(disk.radius), innerRadius(disk.innerRadius),
     phiMax(disk.phiMax)
{

}

optix::Geometry CudaDisk::setupGeometry()
{
    optix::Geometry geomtry=gContext->createGeometry();
    geomtry->setPrimitiveCount(1);
    geomtry->setIntersectionProgram(progIntersection);
    geomtry->setBoundingBoxProgram(progBoundingBox);

    geomtry["height"]->setFloat(height);
    geomtry["radius"]->setFloat(radius);
    geomtry["innerRadius"]->setFloat(innerRadius);
    geomtry["phiMax"]->setFloat(phiMax);

    return geomtry;
}

CUDASHAPE_INIT(disk, CudaDisk);
/*void CudaDisk::init()
{
    const string triangleMeshPTX=ptxpath("util", "cudatrianglemesh.cu");
    progBoundingBox = gContext->createProgramFromPTXFile(triangleMeshPTX, "trianglemesh_bounds" );
    progIntersection = gContext->createProgramFromPTXFile(triangleMeshPTX, "trianglemesh_intersect" );
}*/
