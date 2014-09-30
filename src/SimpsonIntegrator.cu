#include "Integrator.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <curand.h>
#include "RandomVariable.hpp"
#include "kernel.hpp"
#include "config.hpp"
#include "cutil.hpp"


SimpsonIntegrator::SimpsonIntegrator() {}
SimpsonIntegrator::~SimpsonIntegrator() {}

///
// Compute DUST for 2 time series.
//
float
SimpsonIntegrator::distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length) {
  size_t tuples_size  = sizeof(float) * ts_length * TUPLE_SIZE;
  size_t dusts_size = sizeof(float) * ts_length;

  // copy ts1, ts2 to memory
  float *tuples, *dusts, *tuples_GPU, *dusts_GPU;
  tuples = (float*)malloc(tuples_size);
  dusts = (float*)malloc(dusts_size);
  checkCudaErrors(cudaMalloc((void**)&tuples_GPU, tuples_size));
  checkCudaErrors(cudaMalloc((void**)&dusts_GPU, dusts_size));

  int idx = 0;
  for (int i = 0; i < ts_length; i++) {
    RandomVariable x = ts1.at(i);
    RandomVariable y = ts2.at(i);

    tuples[idx]   = (float)x.distribution;
    tuples[idx + 1] = x.observation;
    tuples[idx + 2] = x.stddev;
    tuples[idx + 3] = (float)y.distribution;
    tuples[idx + 4] = y.observation;
    tuples[idx + 5] = y.stddev;
    idx += TUPLE_SIZE;
  }

  checkCudaErrors(cudaMemcpy(tuples_GPU,
                             tuples,
                             tuples_size,
                             cudaMemcpyHostToDevice));

  // call kernel
  g_distance_simpson_kernel<<< ts_length, TPB >>>(tuples_GPU, dusts_GPU, 49152);

  checkCudaErrors(cudaMemcpy(dusts,
                             dusts_GPU,
                             dusts_size,
                             cudaMemcpyDeviceToHost));

  float dust_sum = 0;
  for (int i = 0; i < ts_length; i++) {
    dust_sum += dusts[i];
  }

  free(tuples);
  free(dusts);
  checkCudaErrors(cudaFree(tuples_GPU));
  checkCudaErrors(cudaFree(dusts_GPU));

  return sqrt(dust_sum);
}

// Match 1 ts to all ts in tsc.
// Repeat Integrator::distance for all combination.
void
SimpsonIntegrator::match_naive(const TimeSeries &ts, const TimeSeriesCollection &tsc) {
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
  std::cout << "\t index: " << i_min
            << ", distance : " << DUST_min << std::endl;
}


// Match 1 ts to all ts in tsc
// Optimized version.
void
SimpsonIntegrator::match(const TimeSeries &ts, const TimeSeriesCollection &tsc) {
  this->prepare_match(ts, tsc);

  g_match_simpson<<< ts_length, TPB >>>(ts_D,
                                        tsc_D,
                                        dusts_D,
                                        ts_length,
                                        ts_num,
                                        1024);

  int i_min;
  float DUST_min;
  this->finish_match(&i_min, &DUST_min);

  std::cout << "matched : " << ts_length << std::endl;
  std::cout << "\t index: " << i_min
            << ", distance: " << DUST_min << std::endl;
}
