# This file is derived from the optix sample. Although a lot of detail has been changed, the basic structure is still based on the SDK sample CMakeList. Please reference the license there for further info
set(VERSION "v 0.0.1")

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )
set(CMAKE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

#Try to find the CUDA install path automatically, so that I don't need to set it over and over again
set(_Optix_INSTALL_SEARCH_DIRS
    "C:/Program Files/NVIDIA Corporation/OptiX SDK 3.0.0"
    "../"
  )
find_path(OptiX_INSTALL_DIR NAMES include/optix.h PATHS ${_Optix_INSTALL_SEARCH_DIRS})


##########
# Setup the Cuda compiler enviroment
# Process our custom setup scripts here.

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# Find at least a 2.3 version of CUDA.
find_package(CUDA 2.3 REQUIRED)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

# Add some useful default arguments to the nvcc flags.  This is an example of how we use
# PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE and following
# block of code will not be executed leaving you free to edit the values as much as you
# wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()

  if (CUDA_VERSION VERSION_LESS "3.0")
    set(flag "--keep")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()
endif(NOT PASSED_FIRST_CONFIGURE)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Force the perforce plugin off for the SDK.
set(USE_PERFORCE OFF CACHE BOOL "This is forced off." FORCE)

##################################################################
# Setup the PTX generating DIR and set the path in config.h macro
# The windows accept the / as path seperator (finally). We do not
# need to do special steps to support (evil) windows

set(CUDA_RENDER_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH "Path to where the samples look for the PTX code.")

set(CUDA_GENERATED_OUTPUT_DIR ${CUDA_RENDER_PTX_DIR})

# Don't know what is this, keep it for safe
# Do a little search, seems C99 drop the definition and gcc follows
if ( USING_GCC AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

set (cuda_render_sublibs CACHE INTERNAL "catch all the cuda render implements" FORCE)
function(add_cuda_render_library target_name)
  # group the files in Visual Studio
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
   CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Create the rules to build the PTX from the CUDA files.
   CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options}
     OPTIONS ${options} )

  # I don't know why, but one time I got a CMAKE generate error about the ${CMAKE_CURRENT_SOURCE_DIR}/${generated_files}.
  # It should be ok, because the CUDA_WRAP_SRCS calls ADD_CUSTOM_COMMAND. Just comment that line if thing goes wrong
  # finally, solve the problem. The generated_files already has full path..
    set (current_src
      ${CMAKE_CURRENT_SOURCE_DIR}/${source_files}
      ${generated_files}
      ${CMAKE_CURRENT_SOURCE_DIR}/${cmake_options}
    )

  add_library(${target_name} STATIC ${current_src})
  # CUDA_ADD_LIBRARY(${target_name} ${ARGN})

  set (cuda_render_sublibs ${cuda_render_sublibs} ${target_name} CACHE INTERNAL "catch all the cuda render implements")

  target_link_libraries (${target_name}
    optix
    optixu  #optixu should be included. It is shame for Nvidia link it inside sutil and not document it!!
    ${optix_rpath}
    )

endfunction()

# setup the global setting for the sublibs
include_directories(${CMAKE_SOURCE_DIR}/pbrt-v2/src)
include_directories(${CMAKE_SOURCE_DIR}/pbrt-v2/src/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${OptiX_INCLUDE}
                    ${OptiX_INCLUDE}/optixu
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CUDA_INCLUDE_DIRS})


add_subdirectory(util)
#add_subdirectory(photon_mapping)
add_subdirectory(simple_render)

add_library(cuda_render
    cudarender.h
    cudarender.cpp
    cudaapi.h
    cudaapi.cpp
    #cudarenderhooks.h
    )
target_link_libraries(cuda_render ${cuda_render_sublibs})
message(${cuda_render_sublibs})

# This copies out dlls into the build directories, so that users no longer need to copy
# them over in order to run the samples.
if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
    set(bit_dest "64")
  else()
    set(bit_dest "")
  endif()
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    get_target_property(loc "util" ${config}_LOCATION)
    if(loc)
      # A little helper function
      function(copy_dll lib)
        get_filename_component(path ${loc} PATH)
        get_filename_component(name ${lib} NAME)
        #message("${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name})
      endfunction()

      # Copy the binary directory into the build
      file(GLOB dlls "${OptiX_INSTALL_DIR}/bin${bit_dest}/*.dll")
      foreach(file ${dlls})
        copy_dll("${file}")
      endforeach()
    else()
      message(WARNING "Unable to find location to copy DLLs into the build")
    endif()
  endforeach()
endif(WIN32)

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")

