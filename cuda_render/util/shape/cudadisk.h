#ifndef cudadisk_h__
#define cudadisk_h__

#include "CudaShape.h"
#include "pbrt.h"
#include "shapes/disk.h"
#include "optix_world.h"
class CudaDisk : public CudaShape{
public:
    CudaDisk(const Disk& disk);

    SPACE intersectionSpace(){return CudaShape::OBJECT_SPACE;}
    optix::Geometry setupGeometry();

    static void init();
    static optix::Program progBoundingBox;
    static optix::Program progIntersection;
protected:
    float height, radius, innerRadius, phiMax;
};
#endif // cudadisk_h__