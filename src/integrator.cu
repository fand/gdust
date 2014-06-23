#include "integrator.hpp"
#include "randomvariable.hpp"
#include "kernel.hpp"
#include "config.hpp"

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
    cudaMalloc((void**)&seq_GPU, seq_size);
    cudaMalloc((void**)&dust_GPU, dust_size);

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

    cudaMemcpy( seq_GPU,
                seq,
                seq_size,
                cudaMemcpyHostToDevice );

    // generate uniform random number on samples_GPU
    float *samples_GPU;
    cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * n * 3);
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * n * 3 );


    // call kernel
    g_distance_kernel<<< n, TPB >>>(seq_GPU, samples_GPU, dust_GPU);

    cudaMemcpy( dust,
                dust_GPU,
                dust_size,
                cudaMemcpyDeviceToHost );


    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }


    cudaFree( seq_GPU );
    cudaFree( dust_GPU );
    cudaFree( samples_GPU );

    free(seq);
    free(dust);

    return sqrt(dist);
}


void
Integrator::match_naive (TimeSeries &ts, TimeSeriesCollection &db)
{
    // Determine the length of time series.
    unsigned int lim = ts.length();

    for (int i=0; i < db.sequences.size(); i++) {
        lim = min(lim, db.sequences[i].length());
    }

    float distance_min = this->distance(ts, db.sequences[0], lim);
    float i_min = 0;
    for (int i=1; i < db.sequences.size(); i++) {
        float d = this->distance(ts, db.sequences[i], lim);
        if (d < distance_min) {
            distance_min = d;
            i_min = i;
        }
    }

    std::cout << "matched : " << lim << std::endl;
    std::cout << "\t index: " << i_min << ", distance : " << distance_min << std::endl;
}


void
Integrator::match (TimeSeries &ts, TimeSeriesCollection &db)
{
    // Determine the length of time series.
    size_t db_num = db.sequences.size();
    unsigned int lim = ts.length();
    for (int i=0; i < db_num; i++) {
        lim = min(lim, db.sequences[i].length());
    }

    // db needs (3 * float * lim * num) bytes = 3*4*150*200
    // can settle on global memory.
    size_t db_size = sizeof(float) * 3 * lim * db_num;
    float *db_CPU, *db_GPU;
    db_CPU = (float*)malloc(db_size);
    cudaMalloc((void**)&db_GPU, db_size);

    size_t DUST_size = sizeof(float) * db_num;
    float *DUST_CPU, *DUST_GPU;
    DUST_CPU = (float*)malloc(DUST_size);
    cudaMalloc((void**)&DUST_GPU, DUST_size);

    int idx = 0;
    for (int i = 0; i < db_num; i++) {
        for (int j = 0; j < lim; j++) {
            RandomVariable x = db.sequences[i].at(j);
            db_CPU[idx++]     = (float)x.distribution;
            db_CPU[idx++] = x.observation;
            db_CPU[idx++] = x.stddev;
        }
    }

    size_t ts_size = DUST_size * 3;
    float *ts_CPU, *ts_GPU;
    ts_CPU = (float*)malloc(ts_size);
    cudaMalloc((void**)&ts_GPU, ts_size);
    for (int i = 0; i < lim; i++) {
        RandomVariable x = ts.at(i);
        ts_CPU[i]     = (float)x.distribution;
        ts_CPU[i + 1] = x.observation;
        ts_CPU[i + 2] = x.stddev;
    }

    cudaMemcpy( db_GPU,
                db_CPU,
                db_size,
                cudaMemcpyHostToDevice );

    cudaMemcpy( ts_GPU,
                ts_CPU,
                ts_size,
                cudaMemcpyHostToDevice );

    float *o1, *o2, *o3;
    cudaMalloc((void**)&o1, sizeof(float) * db_num);
    cudaMalloc((void**)&o2, sizeof(float) * db_num);
    cudaMalloc((void**)&o3, sizeof(float) * db_num);


    // generate uniform random number on samples_GPU
    float *samples_GPU;
    cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * lim * 3);
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * lim * 3 );


    // DO THE STUFF
    g_match<<< lim, TPB >>>(ts_GPU, db_GPU, DUST_GPU, lim, db_num, o1, o2, o3, samples_GPU);


    cudaMemcpy( DUST_CPU,
                DUST_GPU,
                DUST_size,
                cudaMemcpyDeviceToHost );


    float DUST_min = DUST_CPU[0];
    int i_min = 0;
    for (int i = 1; i < db_num; i++) {
        float d = DUST_CPU[i];
        if (d < DUST_min) {
            DUST_min = d;
            i_min = i;
        }
    }

    // std::cout << "db_num : " << db_num << std::endl;
    // std::cout << "lim : " << lim << std::endl;
    std::cout << "matched : " << lim << std::endl;
    std::cout << "\t index: " << i_min << ", DUST : " << DUST_min << std::endl;

    free(db_CPU);
    cudaFree(db_GPU);
    free(DUST_CPU);
    cudaFree(DUST_GPU);
    free(ts_CPU);
    cudaFree(ts_GPU);
    cudaFree(o1);
    cudaFree(o2);
    cudaFree(o3);
    cudaFree(samples_GPU);
}
