#include "cudashape.h"
#include "cudatrianglemesh.h"
#include "shapes/sphere.h"
#include "cudasphere.h"
void CudaShape::init()
{
    CudaTriangleMesh::init();
    CudaSphere::init();
}

CudaShape*
CudaShape::CreateCudaShape(const string& name, Reference<Shape>& shape)
{
    if (name=="trianglemesh"){
        CudaTriangleMesh* ctm=new CudaTriangleMesh(shape);
        return ctm;
    }else if(name=="sphere"){
        CudaShape* csp=new CudaSphere(shape);
        return csp;
    }else{
        return nullptr;
    }
}

optix::Geometry CudaShape::setupGeometry()
{
    Severe("setupGeometry function not implemented in current shape");
    return optix::Geometry(); //die at last statement
}

optix::Transform CudaShape::setupTransform()
{
    Severe("setupTransform function not implemented in current shape");
    return optix::Transform(); //die at last statement
}
