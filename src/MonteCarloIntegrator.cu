#include "Integrator.hpp"
#include <iostream>
#include <math.h>
#include <curand.h>
#include "RandomVariable.hpp"
#include "kernel.hpp"
#include "config.hpp"
#include "cutil.hpp"


MonteCarloIntegrator::MonteCarloIntegrator()
{
    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );
}

MonteCarloIntegrator::~MonteCarloIntegrator()
{
    curandDestroyGenerator( *(this->gen) );
}


//!
// Compute DUST for 2 time series.
//
float
MonteCarloIntegrator::distance (TimeSeries &ts1, TimeSeries &ts2, int ts_length)
{
    size_t tuples_size  = sizeof(float) * ts_length * TUPLE_SIZE;
    size_t dusts_size = sizeof(float) * ts_length;

    // copy ts1, ts2 to memory
    float *tuples, *dusts, *tuples_GPU, *dusts_GPU;
    tuples  = (float*)malloc(tuples_size);
    dusts = (float*)malloc(dusts_size);
    checkCudaErrors(cudaMalloc((void**)&tuples_GPU, tuples_size));
    checkCudaErrors(cudaMalloc((void**)&dusts_GPU, dusts_size));

    int idx = 0;
    for (int i = 0; i < ts_length; i++) {
        RandomVariable x = ts1.at(i);
        RandomVariable y = ts2.at(i);

        tuples[idx]   = (float)x.distribution;
        tuples[idx+1] = x.observation;
        tuples[idx+2] = x.stddev;
        tuples[idx+3] = (float)y.distribution;
        tuples[idx+4] = y.observation;
        tuples[idx+5] = y.stddev;
        idx += TUPLE_SIZE;
    }

    checkCudaErrors(cudaMemcpy( tuples_GPU,
                                tuples,
                                tuples_size,
                                cudaMemcpyHostToDevice ));

    // generate uniform random number on samples_GPU
    float *samples_GPU;
    checkCudaErrors(cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * ts_length * 3));
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * ts_length * 3 );


    // call kernel
    g_distance_kernel<<< ts_length, TPB >>>(tuples_GPU, samples_GPU, dusts_GPU);

    checkCudaErrors(cudaMemcpy( dusts,
                                dusts_GPU,
                                dusts_size,
                                cudaMemcpyDeviceToHost ));

    float dust_sum = 0;
    for (int i = 0; i < ts_length; i++) {
      dust_sum += dusts[i];
    }

    free(tuples);
    free(dusts);
    checkCudaErrors(cudaFree( tuples_GPU ));
    checkCudaErrors(cudaFree( dusts_GPU ));
    checkCudaErrors(cudaFree( samples_GPU ));

    return sqrt(dust_sum);
}

// Match 1 ts to all ts in tsc.
// Repeat Integrator::distance for all combination.
void
MonteCarloIntegrator::match_naive (TimeSeries &ts, TimeSeriesCollection &tsc)
{
    // Determine the length of time series.
    unsigned int ts_length = min(ts.length(), tsc.length_min());
    for (int i = 0; i < tsc.sequences.size(); i++) {
        ts_length = min(ts_length, tsc.sequences[i].length());
    }

    float DUST_min;
    float i_min;
    for (int i = 0; i < tsc.sequences.size(); i++) {
        float DUST = this->distance(ts, tsc.sequences[i], ts_length);
        if (DUST < DUST_min || i == 0) {
            DUST_min = DUST;
            i_min = i;
        }
    }

    std::cout << "matched : " << ts_length << std::endl;
    std::cout << "\t index: " << i_min << ", distance : " << DUST_min << std::endl;
}


// Match 1 ts to all ts in tsc
// Optimized version.
void
MonteCarloIntegrator::match (TimeSeries &ts, TimeSeriesCollection &tsc)
{
    this->prepare_match(ts, tsc);

    // Generate uniform random number on samples_GPU.
    size_t samples_num = INTEGRATION_SAMPLES * ts_length * ts_num * 3;
    checkCudaErrors(cudaMalloc( (void**)&(samples_D), sizeof(float) * samples_num));
    curandGenerateUniform( *(this->gen), samples_D, samples_num);

    g_match<<< ts_length, TPB >>>( ts_D,
                                   tsc_D,
                                   dusts_D,
                                   ts_length,
                                   ts_num,
                                   this->samples_D );

    int i_min;
    float DUST_min;
    this->finish_match(&i_min, &DUST_min);
    checkCudaErrors(cudaFree(samples_D));

    std::cout << "matched : " << ts_length << std::endl;
    std::cout << "\t index: " << i_min << ", distance: " << DUST_min << std::endl;
}
