# Prerequest library
# Boost
#set(Boost_DEBUG TRUE)
# I am not using the Boost or opengl now, keep them here in case I need them
#set(Boost_USE_STATIC_LIBS ON) # This is needed because I only compile the static library
#find_package(Boost COMPONENTS program_options REQUIRED)

#if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIR})
#    link_directories(${Boost_LIBRARY_DIRS})
#    add_definitions( "-DHAS_BOOST" )
#else()
#    message(SEND_ERROR "Boost not found by CMAKE")
#endif()

# OPENGL
#if(NOT OPENGL_FOUND)
#    message(SEND_ERROR "OPENGL not found by CMAKE")
#endif()

# GLUT
#if(GLUT_FOUND)
#    include_directories(${GLUT_INCLUDE_DIR})
#    add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)
#else()
#    message(SEND_ERROR "GLUT not found by CMAKE")
#endif()

add_cuda_render_library(photon_mapping
    #Hook.cpp
    photonmappingrenderer.cpp
    photonmappingrenderer.h
    raytracing.cu
    photontracing.cu
    gathering.cu
    photonmapping.h
)

target_link_libraries(photon_mapping util)
#target_link_libraries(photon_mapping ${Boost_LIBRARIES})