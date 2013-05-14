#include "cudashape.h"
#include "cudatrianglemesh.h"
#include "shapes/sphere.h"
#include "cudasphere.h"
#include "cudadisk.h"
void CudaShape::init()
{
    CudaTriangleMesh::init();
    CudaSphere::init();
    CudaDisk::init();
}

CudaShape*
CudaShape::CreateCudaShape(const string& name, Reference<Shape>& shape)
{
    if (name=="trianglemesh"){
        CudaTriangleMesh* ctm=new CudaTriangleMesh(shape);
        return ctm;
    }else if(name=="sphere"){
        CudaSphere* csp=new CudaSphere(shape);
        return csp;
    }else if(name=="disk"){
        CudaDisk* cd=new CudaDisk(shape);
        return cd;
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
