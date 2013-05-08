add_cuda_render_cuda_library(random
    cudarandom.cpp
    cudarandom.h
)

target_link_libraries(random ${CUDA_curand_LIBRARY})
