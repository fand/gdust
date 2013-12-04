#include "integrator.hpp"
#include "randomvariable.hpp"
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
}




float Integrator::distance( TimeSeries &ts1, TimeSeries &ts2, int n )
{
    size_t seq_size = sizeof(float) * n * PARAM_SIZE;
    size_t dust_size = sizeof(float) * n;
    
    // dispatch seq to memory
    float *seq, *dust, *seq_GPU, *dust_GPU;
    seq = (float*)malloc(seq_size);
    dust = (float*)malloc(dust_size);
    // CUDA_SAFE_CALL(cudaMalloc((void**)&seq_GPU, seq_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dust_GPU, dust_size));

    for( int i = 0; i < n; i++ )
    {
        RandomVariable x = ts1.at(i);
        RandomVariable y = ts2.at(i);
        
        int j = PARAM_SIZE * i;
        seq[j] = (float)x.distribution;
        seq[j+1] = x.observation;
        seq[j+2] = x.stddev;
        seq[j+3] = (float)y.distribution;
        seq[j+4] = y.observation;
        seq[j+5] = y.stddev;
    }

    copyToConst(seq, seq_size);

    // generate uniform random number on in_GPU
    float *in;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&in, dust_size * INTEGRATION_SAMPLES * 3) );    
    curandGenerateUniform( *(this->gen), in, INTEGRATION_SAMPLES * n * 3 );

    // cudaMemcpyToSymbol(seq_const, seq, seq_size, 0, cudaMemcpyHostToDevice);
    // cudaGetSymbolAddress((void**)&seq_GPU, seq_const);

    // CUDA_SAFE_CALL( cudaMemcpy( seq_GPU,
    //                             seq,
    //                             seq_size,
    //                             cudaMemcpyHostToDevice ) );

    // call kernel
//    distance_kernel<<< n, TPB >>>(seq_const, this->in_GPU, dust_GPU);
    distance_kernel<<< n, TPB >>>(this->in_GPU, dust_GPU);    

    CUDA_SAFE_CALL( cudaMemcpy( dust,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ) );


    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }


//    CUDA_SAFE_CALL( cudaFree( seq_GPU ) );
    CUDA_SAFE_CALL( cudaFree( dust_GPU ) );
    CUDA_SAFE_CALL( cudaFree( in ) );
    
    free(seq);
    free(dust);
    
    return sqrt(dist);
}

