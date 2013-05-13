
#ifndef cudasphere_h__
#define cudasphere_h__

#include "CudaShape.h"
#include "pbrt.h"
#include "optixpp_namespace.h"
#include "shapes\sphere.h"
class CudaSphere : public CudaShape{
public:
    CudaSphere(const Reference<Shape>& shape);
    SPACE intersectionSpace(){return CudaShape::OBJECT_SPACE;}
    optix::Transform setupTransform();

    static void init();
    static optix::Program progBoundingBox;
    static optix::Program progIntersection;
protected:
    const Reference<Shape>& shape;
};
#endif // cudasphere_h__
