cmake_minimum_required(VERSION 2.6)
project(AdvancedRayTracing)
option(USE_CUDA_RENDER "Enable the cuda render" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# This is copied from pbrt visual studio 2010 solution
# In other platform, the scon script will find the definition. But for windows,
# it is hard coded, and the code from the 3rdparty will be used instead of 
# trying to find the library from system path.
# Keep it here in case I need to port back to linux

add_definitions( -DPBRT_PROBES_NONE -DPBRT_HAS_OPENEXR -DZLIB_WINAPI -D_CRT_SECURE_NO_WARNINGS )

add_subdirectory(cuda_render)
add_subdirectory(pbrt-v2/src)
