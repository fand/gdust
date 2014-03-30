#include "integrator.hpp"
#include "kernel.hpp"

#include <math.h>
#include <fstream>

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
    cudaFree( this->in_GPU );
    cudaFree( this->out_GPU );
    cudaFree( this->param_GPU );
    cudaFree( this->sum_GPU );
    

    cudaFree( this->out1_GPU );
    cudaFree( this->out2_GPU );
    cudaFree( this->out3_GPU );  
    cudaFree( this->sum1_GPU );
    cudaFree( this->sum2_GPU );
    cudaFree( this->sum3_GPU );
    cudaFree( this->answer_GPU );

    
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

    cudaMalloc( (void**)&(this->in_GPU), size );
    cudaMalloc( (void**)&(this->out_GPU), size );
    cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * PARAM_SIZE );
    cudaMalloc( (void**)&(this->sum_GPU), sizeof(float) * BPG );
    
    cudaMalloc( (void**)&(this->out1_GPU), size);
    cudaMalloc( (void**)&(this->out2_GPU), size);
    cudaMalloc( (void**)&(this->out3_GPU), size);
    cudaMalloc( (void**)&(this->sum1_GPU), sizeof(float) );
    cudaMalloc( (void**)&(this->sum2_GPU), sizeof(float) );
    cudaMalloc( (void**)&(this->sum3_GPU), sizeof(float) );
    cudaMalloc( (void**)&(this->answer_GPU), sizeof(float) );
    
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


    cudaMemcpy( this->param_GPU,
                param,
                sizeof(float) * PARAM_SIZE,    // param size
                cudaMemcpyHostToDevice );
    
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

    cudaMemcpy( this->sum,
                this->sum_GPU,
                sizeof(float) * BPG,
                cudaMemcpyDeviceToHost );

    float tes = 0;
    for (int i=0; i < BPG; i++) {
        tes += sum[i];
    }

    return (tes / calls) * (range_max - range_min);
}



float Integrator::phi( float *param )
{

    cudaMemcpy( this->param_GPU,
                param,
                sizeof(float) * PARAM_SIZE,    // param size
                cudaMemcpyHostToDevice );

    // exec!
    dim3 blocks( BPG, 1, 1 );
    dim3 threads( TPB, 1, 1 );

    phi_kernel<<< blocks, threads >>>( param_GPU,
                                       this->in_GPU,
                                       this->answer_GPU );


    float answer = 0;
    cudaMemcpy( &answer,
                this->answer_GPU,
                sizeof(float),
                cudaMemcpyDeviceToHost );
    return answer;
}
