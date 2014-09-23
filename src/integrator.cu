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
    checkCudaErrors(cudaMalloc((void**)&seq_GPU, seq_size));
    checkCudaErrors(cudaMalloc((void**)&dust_GPU, dust_size));

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

    checkCudaErrors(cudaMemcpy( seq_GPU,
                                seq,
                                seq_size,
                                cudaMemcpyHostToDevice ));

    // generate uniform random number on samples_GPU
    float *samples_GPU;
    checkCudaErrors(cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * n * 3));
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * n * 3 );


    // call kernel
    g_distance_kernel<<< n, TPB >>>(seq_GPU, samples_GPU, dust_GPU);

    checkCudaErrors(cudaMemcpy( dust,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ));


    float dist = 0;
    for (int i=0; i < n; i++) {
        dist += dust[i];
    }


    checkCudaErrors(cudaFree( seq_GPU ));
    checkCudaErrors(cudaFree( dust_GPU ));
    checkCudaErrors(cudaFree( samples_GPU ));

    free(seq);
    free(dust);

    return sqrt(dist);
}


void
Integrator::match_naive (TimeSeries &ts, TimeSeriesCollection &db)
{
    // Determine the length of time series.
    unsigned int lim = ts.length();

    for (int i = 0; i < db.sequences.size(); i++) {
        lim = min(lim, db.sequences[i].length());
    }

    float DUST_min;
    float i_min;

    for (int i = 0; i < db.sequences.size(); i++) {
        float DUST = this->distance(ts, db.sequences[i], lim);

        // std::cout << "CPU " << i << ": " << DUST << std::endl;

        if (DUST < DUST_min || i == 0) {
            DUST_min = DUST;
            i_min = i;
        }
    }

    std::cout << "matched : " << lim << std::endl;
    std::cout << "\t index: " << i_min << ", distance : " << DUST_min << std::endl;
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

    // db needs (3 * float * lim * db_num) bytes =~ 3*4*150*200
    // can settle on global memory.
    size_t db_size = sizeof(float) * 3 * db_num * lim;
    float *db_CPU, *db_GPU;
    db_CPU = (float*)malloc(db_size);
    checkCudaErrors(cudaMalloc((void**)&db_GPU, db_size));

    size_t dust_size = sizeof(float) * db_num * lim;
    float *dust_CPU, *dust_GPU;
    dust_CPU = (float*)malloc(dust_size);
    checkCudaErrors(cudaMalloc((void**)&dust_GPU, dust_size));

    int idx = 0;
    for (int i = 0; i < lim; i++) {
        for (int j = 0; j < db_num; j++) {
            RandomVariable x = db.sequences[j].at(i);
            db_CPU[idx++] = (float)x.distribution;
            db_CPU[idx++] = x.observation;
            db_CPU[idx++] = x.stddev;
        }
    }

    size_t ts_size = sizeof(float) * lim * 3;
    float *ts_CPU, *ts_GPU;
    ts_CPU = (float*)malloc(ts_size);
    checkCudaErrors(cudaMalloc((void**)&ts_GPU, ts_size));
    idx = 0;
    for (int i = 0; i < lim; i++) {
        RandomVariable x = ts.at(i);
        ts_CPU[idx++] = (float)x.distribution;
        ts_CPU[idx++] = x.observation;
        ts_CPU[idx++] = x.stddev;
    }

    checkCudaErrors(cudaMemcpy( db_GPU,
                                db_CPU,
                                db_size,
                                cudaMemcpyHostToDevice ));

    checkCudaErrors(cudaMemcpy( ts_GPU,
                                ts_CPU,
                                ts_size,
                                cudaMemcpyHostToDevice ));

    // generate uniform random number on samples_GPU
    float *samples_GPU;
    checkCudaErrors(cudaMalloc( (void**)&samples_GPU, sizeof(float) * INTEGRATION_SAMPLES * lim * db_num * 3));
    curandGenerateUniform( *(this->gen), samples_GPU, INTEGRATION_SAMPLES * lim * db_num * 3);


    // DO THE STUFF
    g_match<<< lim, TPB >>>( ts_GPU,
                             db_GPU,
                             dust_GPU,
                             lim,
                             db_num,
                             samples_GPU );

    checkCudaErrors(cudaMemcpy( dust_CPU,
                                dust_GPU,
                                dust_size,
                                cudaMemcpyDeviceToHost ));

    float DUST_min;
    int i_min = 0;

    for (int i = 0; i < db_num; i++) {
        float dist = 0;
        for (int j = 0; j < lim; j++) {
            dist += dust_CPU[db_num * j + i];
        }

        float DUST = sqrt(dist);
        // std::cout << "GPU " << i << ": " << DUST << std::endl;
        if (DUST < DUST_min || i == 0) {
            DUST_min = DUST;
            i_min = i;
        }
    }

    // std::cout << "db_num : " << db_num << std::endl;
    // std::cout << "lim : " << lim << std::endl;
    std::cout << "matched : " << lim << std::endl;
    std::cout << "\t index: " << i_min << ", distance: " << DUST_min << std::endl;

    free(db_CPU);
    free(dust_CPU);
    free(ts_CPU);
    checkCudaErrors(cudaFree(db_GPU));
    checkCudaErrors(cudaFree(dust_GPU));
    checkCudaErrors(cudaFree(ts_GPU));
    checkCudaErrors(cudaFree(samples_GPU));
}
