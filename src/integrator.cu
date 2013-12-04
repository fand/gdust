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
    curandDestroyGenerator( *(this->gen) );
}


Integrator::Integrator()
{
    size_t size = sizeof(float) * INTEGRATION_SAMPLES;
    
    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );
}


float
Integrator::distance (TimeSeries &ts1, TimeSeries &ts2, int n)
{
    size_t seq_size  = sizeof(float) * n * PARAM_SIZE;
    size_t dust_size = sizeof(float) * n;
    
    // copy ts1, ts2 to memory
    float *seq, *dust, *seq_GPU, *dust_GPU;
    seq  = (float*)malloc(seq_size);
    dust = (float*)malloc(dust_size);
    CUDA_SAFE_CALL(cudaMalloc((void**)&seq_GPU, seq_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dust_GPU, dust_size));

    for (int i = 0; i < n; i++) {
        RandomVariable x = ts1.at(i);
        RandomVariable y = ts2.at(i);
        
        int j = PARAM_SIZE * i;
        seq[j]   = (float)x.distribution;
        seq[j+1] = x.observation;
        seq[j+2] = x.stddev;
        seq[j+3] = (float)y.distribution;
        seq[j+4] = y.observation;
        seq[j+5] = y.stddev;
    }
    
    CUDA_SAFE_CALL( cudaMemcpy( seq_GPU,
                                seq,
                                seq_size,
                                cudaMemcpyHostToDevice ) );
    
    // generate uniform random number on samples_GPU
    float *samples_GPU;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * n * 3) );
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * n * 3 );

    
    // call kernel
    g_distance_kernel<<< n, TPB >>>(seq_GPU, samples_GPU, dust_GPU);

    CUDA_SAFE_CALL( cudaMemcpy( dust,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ) );


    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }


    CUDA_SAFE_CALL( cudaFree( seq_GPU ) );
    CUDA_SAFE_CALL( cudaFree( dust_GPU ) );
    CUDA_SAFE_CALL( cudaFree( samples_GPU ) );
    
    free(seq);
    free(dust);
    
    return sqrt(dist);
}
