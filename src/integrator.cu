#include "integrator.hpp"
#include "randomvariable.hpp"
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
#define TPB 256
#define BPG 192


Integrator::~Integrator()
{
     cudaFree( this->in_GPU ) ;
     cudaFree( this->out_GPU );
     cudaFree( this->param_GPU ) ;
     cudaFree( this->sum_GPU ) ;


     cudaFree( this->out1_GPU ) ;
     cudaFree( this->out2_GPU ) ;
     cudaFree( this->out3_GPU ) ;
     cudaFree( this->sum1_GPU ) ;
     cudaFree( this->sum2_GPU ) ;
     cudaFree( this->sum3_GPU ) ;
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

    cudaMalloc( (void**)&(this->in_GPU), size ) ;
    cudaMalloc( (void**)&(this->out_GPU), size ) ;
    cudaMalloc( (void**)&(this->param_GPU), sizeof(float) * PARAM_SIZE ) ;
    cudaMalloc( (void**)&(this->sum_GPU), sizeof(float) * BPG ) ;

    cudaMalloc( (void**)&(this->out1_GPU), size) ;
    cudaMalloc( (void**)&(this->out2_GPU), size) ;
    cudaMalloc( (void**)&(this->out3_GPU), size) ;
    cudaMalloc( (void**)&(this->sum1_GPU), sizeof(float) ) ;
    cudaMalloc( (void**)&(this->sum2_GPU), sizeof(float) ) ;
    cudaMalloc( (void**)&(this->sum3_GPU), sizeof(float) ) ;
    cudaMalloc( (void**)&(this->answer_GPU), sizeof(float) ) ;

    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );

    // generate uniform random number on in_GPU
    curandGenerateUniform( *(this->gen), this->in_GPU, INTEGRATION_SAMPLES );
}




float Integrator::distance( TimeSeries &ts1, TimeSeries &ts2, int n )
{
    size_t seq_size = sizeof(float) * n * PARAM_SIZE;
    size_t dust_size = sizeof(float) * n;

    // dispatch seq to memory
    int *h_xy_dist, *d_xy_dist;
    float *dust, *dust_GPU;
    float4 *seq, *seq_GPU;
    seq = (float4 *) malloc(sizeof(float4) * n);
    dust = (float*)malloc(dust_size);
    h_xy_dist = (int *) malloc(sizeof(int) * n);
    cudaMalloc((void **) &d_xy_dist, sizeof(int) * n);
    cudaMalloc((void **) &seq_GPU, sizeof(float4) * n);
    cudaMalloc((void **) &dust_GPU, dust_size);

    for( int i = 0; i < n; i++ )
    {
        RandomVariable x = ts1.at(i);
        RandomVariable y = ts2.at(i);

        h_xy_dist[i] = (x.distribution << 2) + y.distribution;
        seq[i] = make_float4(x.observation, x.stddev, y.observation, y.stddev);
    }

    cudaMemcpy(d_xy_dist, h_xy_dist, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(seq_GPU, seq, sizeof(float4) * n, cudaMemcpyHostToDevice);

    // call kernel
    distance_kernel<<< n, TPB >>>(this->in_GPU, dust_GPU, seq_GPU, d_xy_dist);

    cudaMemcpy(dust, dust_GPU, dust_size, cudaMemcpyDeviceToHost);

    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }

    cudaFree( seq_GPU );
    cudaFree( dust_GPU );
    cudaFree(d_xy_dist);

    free(h_xy_dist);
    free(seq);
    free(dust);

    return sqrt(dist);
}

