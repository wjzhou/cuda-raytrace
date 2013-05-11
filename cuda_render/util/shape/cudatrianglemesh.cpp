#include "cudatrianglemesh.h"
#include <pbrt.h>
#include "cudarender.h"
#include "util/util.h"
#include "optixu_math_namespace.h"


optix::Program CudaTriangleMesh::progBoundingBox;
optix::Program CudaTriangleMesh::progIntersection;

CudaTriangleMesh::CudaTriangleMesh(TriangleMesh && tm)
    :TriangleMesh(std::move(tm))
{
}

optix::Geometry CudaTriangleMesh::setupGeometry()
{
    optix::Geometry geomtry=gContext->createGeometry();
    geomtry->setPrimitiveCount(ntris);
    geomtry->setIntersectionProgram(progIntersection);
    geomtry->setBoundingBoxProgram(progBoundingBox);

    bVertices=gContext->createBuffer(RT_BUFFER_INPUT,
        RT_FORMAT_FLOAT3, nverts);
    optix::float3* pVertics=static_cast<optix::float3*>(bVertices->map());

    /* TriangleMesh has already transform the points into world coordinator
     * Can not use memcpy here because Point type is not POD*/
    for (int i=0; i< nverts; ++i){
        pVertics[i]=optix::make_float3(p[i].x, p[i].y, p[i].z);
    }
    bVertices->unmap();
    geomtry["bVertices"]->setBuffer(bVertices);

    bIndices=gContext->createBuffer(RT_BUFFER_INPUT,
        RT_FORMAT_INT3, ntris);
    optix::int3* pIndices=static_cast<optix::int3*>(bIndices->map());
    for (int i=0; i<ntris; ++i){
        pIndices[i]=optix::make_int3(vertexIndex[3*i],
            vertexIndex[3*i+1], vertexIndex[3*i+2]);
    }
    bIndices->unmap();
    geomtry["bIndices"]->setBuffer(bIndices);

    if(n){
        bNs=gContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nverts);
        optix::float3* pNs=static_cast<optix::float3*>(bNs->map());
        for (int i=0; i<nverts; ++i){
            pNs[i]=optix::make_float3(n[i].x, n[i].y, n[i].z);
        }
        bNs->unmap();
    }else{
        bNs=gContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
    }
    geomtry["bNormals"]->setBuffer(bNs);

    if(uvs){
        bUvs=gContext->createBuffer(RT_BUFFER_INPUT,
            RT_FORMAT_FLOAT2, nverts);

        optix::float2* pUvs=static_cast<optix::float2*>(bUvs->map());
        for (int i=0; i<nverts; ++i){
            pUvs[i]=optix::make_float2(uvs[2*i], uvs[2*i+1]);
        }
        bUvs->unmap();
    }else{
        bUvs=gContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
    }
    geomtry["bUvs"]->setBuffer(bUvs);

    return geomtry;
}

void
CudaTriangleMesh::init()
{
    const string triangleMeshPTX=ptxpath("util", "cudatrianglemesh.cu");
    progBoundingBox = gContext->createProgramFromPTXFile(triangleMeshPTX, "trianglemesh_bounds" );
    progIntersection = gContext->createProgramFromPTXFile(triangleMeshPTX, "trianglemesh_intersect" );

}
