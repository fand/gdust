#include "integrator.hpp"
#include "kernel.hpp"

#include <math.h>
#include <fstream>

#include <cutil.h>
#include <curand.h>


#define VERYSMALL 1E-20
#define SQRT3 1.73205081

#define RANGE_VALUE SQRT3*10
#define INTEG_RANGE_MAX 16
#define INTEG_RANGE_MIN -16

#define BLOCK_SIZE 500



Integrator::~Integrator()
{
    CUDA_SAFE_CALL( cudaFree( this->in_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->out_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->param_GPU ) );
    free(this->in);
    free(this->out);

    curandDestroyGenerator( *(this->gen) );
}


Integrator::Integrator()
{
    size_t size = sizeof(float) * INTEGRATION_SAMPLES;
    
    this->in = (float *)malloc(size);
    this->out = (float *)malloc(size);

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->in_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * 8 ) );

    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );
}



float Integrator::integrate( int fnum, float *param )
{
    float range_min = INTEG_RANGE_MIN;    // -16
    float range_max = INTEG_RANGE_MAX;    // +16
    float result;

    int calls = INTEGRATION_SAMPLES;    // 50000

    size_t size_array = sizeof(float) * calls;

    // generate uniform random number on in_GPU
    curandGenerateUniform( *(this->gen), this->in_GPU, calls );
    // curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_MTGP32 );
    // curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );
    // curandGenerateUniform( gen, this->in_GPU, calls );

    CUDA_SAFE_CALL( cudaMemcpy( this->param_GPU, param, sizeof(float) * 8, cudaMemcpyHostToDevice ) ); // param size
    
    // exec!
    dim3 blocks( calls / BLOCK_SIZE, 1, 1 );
    dim3 threads( BLOCK_SIZE, 1, 1 );

    integrate_kernel<<< blocks, threads >>>( this->param_GPU,
                                             fnum,
                                             this->in_GPU,
                                             this->out_GPU );
    
    CUDA_SAFE_CALL( cudaMemcpy( this->out,
                                this->out_GPU,
                                size_array,
                                cudaMemcpyDeviceToHost ) );
    
    float sum = 0.0;
    for ( int i = 0; i < calls; i++ ) {
        sum += this->out[i];
    }

    result = ( sum / (float)calls ) * ( range_max - range_min );

    return result;
}
