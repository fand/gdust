#include "integrator.hpp"
#include "kernel.hpp"

#include <math.h>
#include <fstream>

#include <cutil.h>
#include <curand.h>

#include <iostream>



#define VERYSMALL 1E-20
#define SQRT3 1.73205081

#define RANGE_VALUE SQRT3*10
#define INTEG_RANGE_MAX 16
#define INTEG_RANGE_MIN -16

#define PARAM_SIZE 8

#define INTEGRATION_SAMPLES 49152
#define TPB 512
#define BPG 96


Integrator::~Integrator()
{
    CUDA_SAFE_CALL( cudaFree( this->in_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->out_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->param_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->sum_GPU ) );
    free(this->in);
    free(this->out);
    free(this->sum);

    curandDestroyGenerator( *(this->gen) );
}


Integrator::Integrator()
{
    size_t size = sizeof(float) * INTEGRATION_SAMPLES;
    
    this->in = (float *)malloc(size);
    this->out = (float *)malloc(size);
    this->sum = (float *)malloc(sizeof(float) * BPG);

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->in_GPU), size * 3 ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * PARAM_SIZE ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum_GPU), sizeof(float) * BPG ) );
    
    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );

}



float Integrator::integrate( int fnum, float *param )
{
    float range_min = INTEG_RANGE_MIN;    // -16
    float range_max = INTEG_RANGE_MAX;    // +16


    int calls = INTEGRATION_SAMPLES;    // 49152


    CUDA_SAFE_CALL(
        cudaMemcpy( this->param_GPU,
                    param,
                    sizeof(float) * PARAM_SIZE,    // param size
                    cudaMemcpyHostToDevice ) );

    
    // generate uniform random number on in_GPU
    curandGenerateUniform( *(this->gen), this->in_GPU, INTEGRATION_SAMPLES * 3 );

    
    // exec!
    dim3 blocks( BPG, 1, 1 );
    dim3 threads( TPB, 1, 1 );

    integrate_kernel<<< blocks, threads >>>( this->param_GPU,
                                             fnum,
                                             this->in_GPU,
                                             this->out_GPU,
                                             range_min,
                                             range_max );
    

    CUDA_SAFE_CALL(
        cudaMemcpy( this->out, this->out_GPU,
                    sizeof(float) * calls, cudaMemcpyDeviceToHost ) );

    float total = 0;
    for (int i = 0; i < calls; i++) {
        total += this->out[i];
    }

    return total * ( range_max - range_min ) / calls;
}
