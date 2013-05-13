#ifndef cudatrianglemesh_h__
#define cudatrianglemesh_h__

#include "CudaShape.h"
#include "pbrt.h"
#include "shapes\trianglemesh.h"
#include "optixpp_namespace.h"
class CudaTriangleMesh: public CudaShape{
public:
    CudaTriangleMesh(const Reference<Shape>& triangleMesh);

    SPACE intersectionSpace(){return CudaShape::GLOBAL_SPACE;}
    optix::Geometry setupGeometry();

    static void init();
    static optix::Program progBoundingBox;
    static optix::Program progIntersection;
protected:
    optix::Buffer bVertices;
    optix::Buffer bUvs;
    optix::Buffer bIndices;
    optix::Buffer bNs;
    const Reference<Shape> shape;
};
#endif // cudatrianglemesh_h__