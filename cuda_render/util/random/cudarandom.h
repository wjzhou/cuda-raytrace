#ifndef random_h__
#define random_h__
#include <memory>
//the curandGenerator_t is a typedef, so forward declaration does not work
//But I do not want to include the "curand" here because it include cuda runtime
//which conflict with optix
//http://stackoverflow.com/questions/836551/forward-declare-a-classs-public-typedef-in-c
//All the five solution is not acceptable
//The fifth will break if cuda change that typedef, and defeats the purpose of using a typedef
//Here, use the Pimpl idiom from boost
//http://www.boost.org/doc/libs/1_47_0/libs/smart_ptr/sp_techniques.html#pimpl

class CudaRandom{
public:
    CudaRandom(unsigned int seed=777);
    void generate(float* d_Rand, unsigned int rand_n);
    ~CudaRandom();
private:
    class impl;
    std::unique_ptr<impl> pimpl_;
};
#endif // random_h__
