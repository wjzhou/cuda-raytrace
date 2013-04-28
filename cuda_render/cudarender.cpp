#include "photon_mapping/PhotonMappingRenderer.h"

#include "cudarender.h"
#include "config.h"
#include "optixpp_namespace.h"
#include "optixu_math_namespace.h"
#include "util/shape/cudashape.h"
#include "util/camera/pbrtcamera.h"
#include "api.h"
#include "simple_render/simplerender.h"


optix::Context gContext;


void CudaRender::init()
{
    gContext=optix::Context::create();
    CudaShape::init();
}

optix::Group CudaRender::assembleNode(
    std::vector<optix::GeometryInstance>& geometryInstances,
    std::vector<optix::Transform>& transforms,
    int nReserveChildn)
{
    optix::Group group=gContext->createGroup();
    group->setAcceleration(gContext->createAcceleration("Sbvh", "Bvh"));

    //setup the geometrygroup(Those intersection in global space)
    if(geometryInstances.size() > 0){
        group->setChildCount(transforms.size()+1);
        optix::GeometryGroup geometryGroup=gContext->createGeometryGroup();
        geometryGroup->setAcceleration(gContext->createAcceleration("Sbvh", "Bvh"));
        geometryGroup->setChildCount(geometryInstances.size());
        for (std::vector<optix::GeometryInstance>::size_type i=0;
            i != geometryInstances.size(); ++i)
        {
            geometryGroup->setChild(i, geometryInstances[i]);
        }
        group->setChild(transforms.size(), geometryGroup);
    }else{
        group->setChildCount(transforms.size());
    }

    // setup the transform(either object instance or shape using
    // object_space intersection)
    if (transforms.size() != 0)
    {
        for (std::vector<optix::Transform>::size_type i=0;
            i != transforms.size(); ++i)
        {
            group->setChild(i, transforms[i]);
        }
    }
    geometryInstances.clear();
    transforms.clear();
    return group;
}

void CudaRender::assembleObject(std::vector<Reference<Primitive> >* key)
{
    if (instanceGeometryInstances.size()+instanceTransforms.size() == 0){
        Warning("empty instance definition");
        return;
    }
    optix::Group group=assembleNode(instanceGeometryInstances, instanceTransforms);
    instanceGroups[key]=group; //no need to detect duplicae, not possible
    lastInstance=nullptr;
}

void CudaRender::objectInstance(std::vector<Reference<Primitive> >* instance,
                                const Transform& tr)
{
     std::map<std::vector<Reference<Primitive> >*, optix::Group>::const_iterator it;
     it=instanceGroups.find(instance);
     if(it == instanceGroups.end()){
         Severe("Instance not found, must our bug, because pbrt has checked already");
     }
     optix::Transform trNode=gContext->createTransform();
     trNode->setMatrix(false, &tr.GetMatrix().m[0][0], &tr.GetInverseMatrix().m[0][0]);
     trNode->setChild(it->second);
     topTransforms.push_back(trNode);
}


CudaRender::CudaRender()
{
    lastInstance=nullptr;
    init();
}

void CudaRender::Render(const Scene *scene)
{
    //finish the graph scene and pass control to sub renderer.
    if(lastInstance != nullptr){
        assembleObject(lastInstance);
    }
    topGroup=assembleNode(topGeometryInstances, topTransforms);
    gContext["top_group"]->set(topGroup);

    CudaCamera* ca=new PbrtCamera();
    renderer->render(scene, this, ca);
}


void CudaRender::createSubRenderer(Sampler *sampler, Camera *camera,
    const ParamSet &params, const std::string& rendername)
{
    if(rendername=="photon_mapping"){
         //renderer=new PhotonMappingRenderer(sampler, camera, params);
    }else{
        renderer=new SimpleRenderer(sampler, camera, params);
    }
}

void CudaRender::createCudaShape(const std::string& name, Reference<Shape>& shape,
    vector<Reference<Primitive> >* currentInstance,
    const Material* kMaterial)
{
    //Last pbrt object define has been end
    if ((lastInstance != nullptr) && (currentInstance != lastInstance)){
        assembleObject(currentInstance);
        lastInstance=currentInstance;
    }

    CudaShape* cs=CudaShape::CreateCudaShape(name, shape);
    if (cs==nullptr){
        Warning("shape:%s not implemented yet", name.c_str());
        return;
    }

    std::vector<optix::GeometryInstance>* resultGeometryInstances=&topGeometryInstances;
    std::vector<optix::Transform>* resultTransforms=&topTransforms;

    if(currentInstance){
        resultGeometryInstances=&instanceGeometryInstances;
        resultTransforms=&instanceTransforms;
    }

    optix::GeometryInstance instance;
    if (cs->intersectionSpace() == CudaShape::GLOBAL_SPACE)
    {
        optix::Geometry geometry=cs->setupGeometry();
        //The optix c++ interface to validate is a joke..
        //It will trigger a segment fault
        if (geometry.get()==nullptr){
            Warning("shape: %s setup geometry fail", name.c_str());
            return;
        }
        instance=gContext->createGeometryInstance();
        instance->setGeometry(geometry);
        geometryInstances.push_back(instance);
        resultGeometryInstances->push_back(instance);
    } else if (cs->intersectionSpace() == CudaShape::OBJECT_SPACE){
        optix::Transform transform=cs->setupTransform();
        instance=transform->getChild<optix::GeometryInstance>();
        geometryInstances.push_back(instance);
        resultTransforms->push_back(transform);
    }

    auto it=materials.find(kMaterial);
    //CudaMaterial*& material=materials[kMaterial];
    CudaMaterial* material;
    if (it==materials.end())
    {
        material=CudaMaterial::createCudaMeteral(kMaterial);
        materials[kMaterial]=material;
    }
    else
    {
        material=(*it).second;
    }
    material->setupMaterial(instance);
    
}

static std::map<std::pair<Transform*,Transform*>, optix::Transform>
    transform2CudaTransformMap;
optix::GeometryGroup transform2CudaGeometryGroup(Transform* o2w, Transform*w2o){
    std::pair<Transform*,Transform*> key(o2w,w2o);
    std::map<std::pair<Transform*,Transform*>, optix::Transform>::const_iterator it;
    it=transform2CudaTransformMap.find(key);
    if (it==transform2CudaTransformMap.end()){
        optix::GeometryGroup group;
        optix::Transform transform;
        transform->setMatrix(false,
            &o2w->GetMatrix().m[0][0], &w2o->GetMatrix().m[0][0]);
        transform->setChild(group);
        transform2CudaTransformMap[key]=transform;
        it=transform2CudaTransformMap.find(key);
    }
    //const Transform& tr=it->second;
    return it->second->getChild<optix::GeometryGroup>();
};