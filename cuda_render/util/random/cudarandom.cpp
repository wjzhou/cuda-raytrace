
#include <stdio.h>
#include "cudarandom.h"
#include <curand.h>
#include "../cuda/helper_cuda.h"

class CudaRandom::impl
{
private:
    impl(impl const &);
    impl & operator=(impl const &);
    curandGenerator_t prngGPU;
    // private data
public:
    impl(unsigned int seed)
    {
        checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
        checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));
    }
    ~impl()
    {
        checkCudaErrors(curandDestroyGenerator(prngGPU));
    }
    void generate(float* d_Rand, unsigned int rand_n)
    {
        checkCudaErrors(curandGenerateUniform(prngGPU, d_Rand, rand_n));
    }
};

CudaRandom::CudaRandom(unsigned int seed)
    :pimpl_(new impl(seed))
{
}

//The unique_ptr need complete class definition for dtor. Thus, this method
//must been defined here
CudaRandom::~CudaRandom()
{
}

void CudaRandom::generate(float* d_Rand, unsigned int rand_n)
{
    pimpl_->generate(d_Rand, rand_n);
}
