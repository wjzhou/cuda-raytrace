#include "cudasample.h"
uint32_t CudaSample::Add2D(uint32_t& num)
{
    num=sampler->RoundSize(num);
    uint32_t offset=Sample2DOffset;
    Sample2DOffset+=num;
    sample->Add2D(num);
    return offset;
}

uint32_t CudaSample::Add1D(uint32_t& num)
{
    num=sampler->RoundSize(num);
    uint32_t offset=Sample1DOffset;
    Sample1DOffset+=num;
    sample->Add1D(num);
    return offset;
}

CudaSample::CudaSample(const Sampler* sampler) 
    :sampler(sampler),
    Sample1DOffset(0), Sample2DOffset(0)
{
    sample=new Sample(nullptr, nullptr, nullptr, nullptr);
}

