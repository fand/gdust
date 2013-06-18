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

#define PARAM_SIZE 6

#define INTEGRATION_SAMPLES 49152
#define TPB 512
#define BPG 96


Integrator::~Integrator()
{
    CUDA_SAFE_CALL( cudaFree( this->in_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->out_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->param_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->sum_GPU ) );
    

    CUDA_SAFE_CALL( cudaFree( this->out1_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->out2_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->out3_GPU ) );    
    CUDA_SAFE_CALL( cudaFree( this->sum1_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->sum2_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->sum3_GPU ) );
    CUDA_SAFE_CALL( cudaFree( this->answer_GPU ) );

    
    free(this->in);
    free(this->out);
    free(this->sum);
    free(this->sum1);
    free(this->sum2);
    free(this->sum3);

    curandDestroyGenerator( *(this->gen) );
}


Integrator::Integrator()
{
    size_t size = sizeof(float) * INTEGRATION_SAMPLES;
    
    this->in = (float *)malloc(size);
    this->out = (float *)malloc(size);
    this->sum = (float *)malloc(sizeof(float) * BPG);
    this->sum1 = (float *)malloc(sizeof(float));
    this->sum2 = (float *)malloc(sizeof(float));
    this->sum3 = (float *)malloc(sizeof(float));    

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->in_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out_GPU), size ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * PARAM_SIZE ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum_GPU), sizeof(float) * BPG ) );
    
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out1_GPU), size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out2_GPU), size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->out3_GPU), size) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum1_GPU), sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum2_GPU), sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->sum3_GPU), sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(this->answer_GPU), sizeof(float) ) );
    
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


    int calls = INTEGRATION_SAMPLES;    // 49152


    CUDA_SAFE_CALL(
        cudaMemcpy( this->param_GPU,
                    param,
                    sizeof(float) * PARAM_SIZE,    // param size
                    cudaMemcpyHostToDevice ) );
    
    // exec!
    dim3 blocks( BPG, 1, 1 );
    dim3 threads( TPB, 1, 1 );

    integrate_kernel<<< blocks, threads >>>( this->param_GPU,
                                             fnum,
                                             this->in_GPU,
                                             this->out_GPU,
                                             range_min,
                                             range_max );

    reduce<float>(calls, TPB, BPG, this->out_GPU, this->sum_GPU);

    CUDA_SAFE_CALL( cudaMemcpy( this->sum,
                                this->sum_GPU,
                                sizeof(float) * BPG,
                                cudaMemcpyDeviceToHost ) );

    float tes = 0;
    for (int i=0; i < BPG; i++) {
        tes += sum[i];
    }

    return (tes / calls) * (range_max - range_min);
}



float Integrator::phi( float *param )
{
    CUDA_SAFE_CALL(
        cudaMemcpy( this->param_GPU,
                    param,
                    sizeof(float) * PARAM_SIZE,    // param size
                    cudaMemcpyHostToDevice ) );

    // exec!
    dim3 blocks( BPG, 1, 1 );
    dim3 threads( TPB, 1, 1 );

    phi_kernel<<< blocks, threads >>>( param_GPU,
                                       this->in_GPU,
                                       this->answer_GPU );


    float answer = 0;
    CUDA_SAFE_CALL( cudaMemcpy( &answer,
                                this->answer_GPU,
                                sizeof(float),
                                cudaMemcpyDeviceToHost ) );
    return answer;
}
