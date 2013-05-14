#ifndef cudadisk_h__
#define cudadisk_h__

#include "CudaShape.h"
#include "pbrt.h"
#include "shapes/disk.h"
#include "optix_world.h"
class CudaDisk : public CudaShape{
public:
    CudaDisk(const Reference<Shape>& shape);

    SPACE intersectionSpace(){return CudaShape::GLOBAL_SPACE;}
    optix::Geometry setupGeometry();

    static void init();
    static optix::Program progBoundingBox;
    static optix::Program progIntersection;
protected:
    const Reference<Shape>& shape;
};
#endif // cudadisk_h__