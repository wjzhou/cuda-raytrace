add_cuda_render_library (util
    util.h
    util.cpp
    util.cu.h
    common.cu.h
    dummy.cu
    shape/cudashape.h
    shape/cudashape.cpp
    shape/cudatrianglemesh.cu
    shape/cudatrianglemesh.h
    shape/cudatrianglemesh.cpp
    camera/camera.cu.h
    camera/pbrtcamera.cu
    camera/pbrtcamera.h
    camera/pbrtcamera.cpp
    sampler/cudasample.h
    sampler/cudasample.cpp
    light/cudalight.h
    light/cudalight.cpp
    light/cudalight.cu.h
    texture/cudatexture.cu.h
    material/cudamaterial.h
    material/cudamaterial.cu.h
    material/cudamaterial.cpp
    )

add_subdirectory(random)