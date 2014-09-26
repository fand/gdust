#include "integrator.hpp"
#include "randomvariable.hpp"
#include "kernel.hpp"
#include "config.hpp"
#include "cutil.hpp"

#include <curand.h>

#include <iostream>
#include <fstream>
#include <math.h>


Integrator::~Integrator()
{
    curandDestroyGenerator( *(this->gen) );
}


Integrator::Integrator()
{
    this->gen = new curandGenerator_t();
    curandCreateGenerator( this->gen, CURAND_RNG_PSEUDO_MTGP32 );
    curandSetPseudoRandomGeneratorSeed( *(this->gen), 1234ULL );
}

/**
 * Compute DUST for 2 time series.
 *
 */
float
Integrator::distance (TimeSeries &ts1, TimeSeries &ts2, int n)
{
    size_t tuples_size  = sizeof(float) * n * TUPLE_SIZE;
    size_t dust_size = sizeof(float) * n;

    // copy ts1, ts2 to memory
    float *tuples, *dust, *tuples_GPU, *dust_GPU;
    tuples  = (float*)malloc(tuples_size);
    dust = (float*)malloc(dust_size);
    checkCudaErrors(cudaMalloc((void**)&tuples_GPU, tuples_size));
    checkCudaErrors(cudaMalloc((void**)&dust_GPU, dust_size));

    int idx = 0;
    for (int i = 0; i < n; i++) {
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
    checkCudaErrors(cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * n * 3));
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * n * 3 );


    // call kernel
    g_distance_kernel<<< n, TPB >>>(tuples_GPU, samples_GPU, dust_GPU);

    checkCudaErrors(cudaMemcpy( dust,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ));

    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }

    free(tuples);
    free(dust);
    checkCudaErrors(cudaFree( tuples_GPU ));
    checkCudaErrors(cudaFree( dust_GPU ));
    checkCudaErrors(cudaFree( samples_GPU ));

    return sqrt(dist);
}

// Match 1 ts to all ts in tsc.
// Repeat Integrator::distance for all combination.
void
Integrator::match_naive (TimeSeries &ts, TimeSeriesCollection &tsc)
{
    // Determine the length of time series.
    unsigned int ts_length = ts.length();

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
Integrator::match (TimeSeries &ts, TimeSeriesCollection &tsc)
{
    // Determine the length of time series.
    size_t ts_num = tsc.sequences.size();
    unsigned int ts_length = ts.length();
    for (int i = 0; i < ts_num; i++) {
        ts_length = min(ts_length, tsc.sequences[i].length());
    }

    // db needs (3 * float * ts_length * ts_num) bytes =~ 3*4*150*200
    float *ts_CPU, *ts_GPU;
    float *tsc_CPU, *tsc_GPU;
    float *dust_CPU, *dust_GPU;
    size_t ts_size = sizeof(float) * ts_length * 3;
    size_t tsc_size = sizeof(float) * ts_num * ts_length * 3;
    size_t dust_size = sizeof(float) * ts_num * ts_length;
    ts_CPU = (float*)malloc(ts_size);
    tsc_CPU = (float*)malloc(tsc_size);
    dust_CPU = (float*)malloc(dust_size);
    checkCudaErrors(cudaMalloc((void**)&ts_GPU, ts_size));
    checkCudaErrors(cudaMalloc((void**)&tsc_GPU, tsc_size));
    checkCudaErrors(cudaMalloc((void**)&dust_GPU, dust_size));

    // Copy & load data.
    int idx = 0;
    for (int i = 0; i < ts_length; i++) {
        RandomVariable x = ts.at(i);
        ts_CPU[idx++] = (float)x.distribution;
        ts_CPU[idx++] = x.observation;
        ts_CPU[idx++] = x.stddev;
    }
    idx = 0;
    for (int i = 0; i < ts_length; i++) {
        for (int j = 0; j < ts_num; j++) {
            RandomVariable x = tsc.sequences[j].at(i);
            tsc_CPU[idx++] = (float)x.distribution;
            tsc_CPU[idx++] = x.observation;
            tsc_CPU[idx++] = x.stddev;
        }
    }
    checkCudaErrors(cudaMemcpy( ts_GPU,
                                ts_CPU,
                                ts_size,
                                cudaMemcpyHostToDevice ));
    checkCudaErrors(cudaMemcpy( tsc_GPU,
                                tsc_CPU,
                                tsc_size,
                                cudaMemcpyHostToDevice ));

    // Generate uniform random number on samples_GPU.
    float *samples_GPU;
    size_t samples_num = INTEGRATION_SAMPLES * ts_length * ts_num * 3;
    checkCudaErrors(cudaMalloc( (void**)&samples_GPU, sizeof(float) * samples_num));
    curandGenerateUniform( *(this->gen), samples_GPU, samples_num);

    // DO THE STUFF
    g_match<<< ts_length, TPB >>>( ts_GPU,
                             tsc_GPU,
                             dust_GPU,
                             ts_length,
                             ts_num,
                             samples_GPU );

    // Return results.
    checkCudaErrors(cudaMemcpy( dust_CPU,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ));

    // Compare DUST & get i for smallest DUST.
    float DUST_min;
    int i_min = 0;
    for (int i = 0; i < ts_num; i++) {
        float dist = 0;
        for (int j = 0; j < ts_length; j++) {
            dist += dust_CPU[ts_num * j + i];
        }

        float DUST = sqrt(dist);
        if (DUST < DUST_min || i == 0) {
            DUST_min = DUST;
            i_min = i;
        }
    }

    std::cout << "matched : " << ts_length << std::endl;
    std::cout << "\t index: " << i_min << ", distance: " << DUST_min << std::endl;

    free(tsc_CPU);
    free(ts_CPU);
    free(dust_CPU);
    checkCudaErrors(cudaFree(tsc_GPU));
    checkCudaErrors(cudaFree(ts_GPU));
    checkCudaErrors(cudaFree(dust_GPU));
    checkCudaErrors(cudaFree(samples_GPU));
}
