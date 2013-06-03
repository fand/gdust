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
#define BLOCK_SIZE 512
#define TPB 512
//#define BLOCK_SIZE 384


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
    this->sum = (float *)malloc(size);

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->in_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * PARAM_SIZE ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum_GPU), size / TPB ) );
    
    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );

    // generate uniform random number on in_GPU
    curandGenerateUniform( *(this->gen), this->in_GPU, INTEGRATION_SAMPLES );
}



float Integrator::integrate( int fnum, float *param )
{
    float range_min = INTEG_RANGE_MIN;    // -16
    float range_max = INTEG_RANGE_MAX;    // +16
    float result;

    int calls = INTEGRATION_SAMPLES;    // 49152

    size_t size_array = sizeof(float) * calls;


    CUDA_SAFE_CALL(
        cudaMemcpy( this->param_GPU,
                    param,
                    sizeof(float) * PARAM_SIZE,    // param size
                    cudaMemcpyHostToDevice ) );
    
    // exec!
    dim3 blocks( calls / BLOCK_SIZE, 1, 1 );
    dim3 threads( BLOCK_SIZE, 1, 1 );

    integrate_kernel<<< blocks, threads >>>( this->param_GPU,
                                             fnum,
                                             this->in_GPU,
                                             this->out_GPU,
                                             range_min,
                                             range_max );


    reduce<float>(calls, TPB, calls/TPB, this->out_GPU, this->sum_GPU);

    CUDA_SAFE_CALL( cudaMemcpy( this->sum,
                                this->sum_GPU,
                                sizeof(float) * calls/TPB,
                                cudaMemcpyDeviceToHost ) );


    float tes = 0;
    for (int i=0; i < calls/TPB; i++) {
        tes += sum[i];
    }
    result = tes * ( range_max - range_min ) / (float)calls;
    
    // float tes = this->sum[0] / calls;
    // result = tes * ( range_max - range_min );

    
/*
    CUDA_SAFE_CALL( cudaMemcpy( this->out,
                                this->out_GPU,
                                size_array,
                                cudaMemcpyDeviceToHost ) );

    float sum = 0.0f;
    for ( int i = 0; i < calls; i++ ) {
        if (this->out[i] == 89898989) {
            std::cout << "OMG p1 or p2 is nan!!" << std::endl;
            exit(0);
        }
        sum += this->out[i];
    }
    result = sum * ( range_max - range_min ) / (float)calls;
*/

    
    return result;
}
