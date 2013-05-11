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
    Shape* pShape=shape.release();
    if (name=="trianglemesh"){
        TriangleMesh* tm=static_cast<TriangleMesh*>(pShape);
        CudaTriangleMesh* ctm=new CudaTriangleMesh (std::move(*tm));
        delete tm;
        shape=ctm;
        return ctm;
    }else if(Sphere* sp=dynamic_cast<Sphere*>(pShape)){
        CudaShape* csp=new CudaSphere(sp);
        shape=pShape;
        return csp;
    }else{
        shape=pShape;
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
