#ifndef cudashape_h__
#define cudashape_h__

#include "optixpp_namespace.h"
#include "memory.h"
/// <summary>
/// CudaShape is a kind of interface class for all the shapes used in cuda render
/// </summary>
/// The main usage of cuda shape is to import the pbrt shape into the optix graph scene.
///    There are two type of shapes base on the intersection coordinate:
///    <list type="number">
///    /lt
///    <item>
/// <term>global_space</term>
/// <description>The shape do the intersection in global space</description>
/// </item>
/// <item>
/// <term>object_space</term>
/// <description>The shape do the intersection in object space</description>
/// </item>
/// Note: the global_space is world space for the top level shapes and the space in
/// the instance for shapes in object instance.
///
///    For these intersect in object space, the function  must return
///    OBJECT_SPACE and the setupTransformNode() must be implemented.
/// On the other hand, for object intersect in global space, GLOBAL_SPACE should be
///    returned and the setupGeometryNode() must be implemented.
struct CudaShape{
public:
    /// This enum indicate which coord space is used for intersection
    /// Most shape of pbrt will use object space while triangle mesh
    /// use the global space.
    enum SPACE {OBJECT_SPACE, GLOBAL_SPACE};
    /// <summary>
    /// SubShape must override this method, caller will use this information
    /// to detemine which setup* method to call.
    /// </summary>
    virtual SPACE intersectionSpace()=0;


    /// <returns>the geometry node</returns>
    virtual optix::Geometry setupGeometry();


    /// <return>the transform node, the geometry node is under transfor node</return>
    virtual optix::Transform setupTransform();

    /// <summary>
    /// This method is called by cudarender and init time. All the sub-type should
    /// register their init in the implement in this method
    /// </summary>
    static void init();

    /// <summary>
    /// This method is called by cudarender to create cuda shape from pbrt shape
    /// </summary>
    /// <remarks>
    /// The pbrt shape may be destroyed and replaced by the sub cuda shape.
    /// (to by-pass the protected access restrict). The caller need to make sure
    /// the shape reference is the only reference to the pbrt shape
    /// </remarks>
    /// <returns>The Cuda shape correspong to the pbrt shape ot NULL if the shape
    /// has not been suppoted yet</returns>
    /// <param name="name">the name of the shape(because lack RTTI info)</param>
    /// <param name="shape">the pbrt shape</param>
    static CudaShape* CreateCudaShape(const std::string& name, Reference<Shape>& shape);
    virtual ~CudaShape(){};
};
#endif // cudashape_h__