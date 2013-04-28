#ifndef cudasample_h__
#define cudasample_h__
#include <core/sampler.h>
/// <summary>
/// The structure of sample used in pbrt is too complicate for GPU
/// this class use a flatted layout, all the returned offset is related
/// to the beginning of random buffer.
/// </summary>
/// 
class CudaSample{
public:
    CudaSample(const Sampler* sampler);
    uint32_t Add1D(uint32_t& num);
    uint32_t Add2D(uint32_t& num);
     Sample* sample;

    const Sampler* sampler;
    uint32_t Sample1DOffset;
    uint32_t Sample2DOffset;
};

#endif // cudasample_h__
