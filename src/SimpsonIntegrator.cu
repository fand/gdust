#include "Integrator.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <curand.h>
#include "RandomVariable.hpp"
#include "config.hpp"
#include "kernel.hpp"
#include "dust_inner.hpp"

#include "gpu.hpp"
#include "cutil.hpp"


// local functions.
__global__ void
distance_kernel(float *tuples_GPU, float *dust_GPU, int division);
__global__ void
match_kernel(float *ts_GPU, float *tsc_GPU, float *dust_GPU,
             size_t ts_length, size_t ts_num, int division);


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
  distance_kernel<<< ts_length, TPB >>>(tuples_GPU, dusts_GPU, 49152);

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

  match_kernel<<< ts_length, TPB >>>(ts_D,
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


// With seq on global memory
__global__ void
distance_kernel(float *tuples_GPU,
                float *dust_GPU,
                int division) {
  float *tuple = tuples_GPU  + blockIdx.x * TUPLE_SIZE;
  float *answer_GPU  = dust_GPU + blockIdx.x;

  float o1 = 0.0f;
  float o2 = 0.0f;
  float o3 = 0.0f;
  __shared__ float sdata1[TPB];
  __shared__ float sdata2[TPB];
  __shared__ float sdata3[TPB];

  float width = RANGE_WIDTH / static_cast<float>(division);

  // MAP PHASE
  // put (f1, f2, f3) into (o1, o2, o3) for all samples
  for (int i = threadIdx.x; i < division; i += blockDim.x) {
    float window_left = width * i + RANGE_MIN;
    o1 += simpson_f1(window_left, width, tuple);
    o2 += simpson_f2(window_left, width, tuple);
    o3 += simpson_f3(window_left, width, tuple);
  }

  // REDUCE PHASE
  // Get sum of (o1, o2, o3) for all threads
  sdata1[threadIdx.x] = o1;
  sdata2[threadIdx.x] = o2;
  sdata3[threadIdx.x] = o3;
  reduceBlock<TPB>(sdata1, sdata2, sdata3);

  if (threadIdx.x == 0) {
    float int1 = sdata1[0];
    float int2 = sdata2[0];
    float int3 = sdata3[0];
    if (int1 < VERYSMALL) int1 = VERYSMALL;
    if (int2 < VERYSMALL) int2 = VERYSMALL;
    if (int3 < 0.0f)      int3 = 0.0f;

    float dust = -log10(int3 / (int1 * int2));
    if (dust < 0.0) { dust = 0.0f; }

    *answer_GPU = dust;
  }
}

//!
// With seq on global memory.
//
__global__ void
match_kernel(float *ts_GPU,
             float *tsc_GPU,
             float *dust_GPU,
             size_t ts_length,
             size_t ts_num,
             int division) {
  int time = blockIdx.x;

  float o1 = 0.0f;
  float o2 = 0.0f;
  float o3 = 0.0f;
  __shared__ float sdata1[TPB];
  __shared__ float sdata2[TPB];
  __shared__ float sdata3[TPB];

  float *dusts = &dust_GPU[ts_num * time];
  float *tsc = &tsc_GPU[ts_num * time * 3];  // TimeSeriesCollection for this block
  float *x  = &ts_GPU[time * 3];             // TODO: compute f1 only once.

  float width = static_cast<float>(RANGE_WIDTH) / division;

  for (int i = 0; i < ts_num; i++) {
    float *y = &tsc[i * 3];
    o1 = o2 = o3 = 0.0f;
    for (int j = threadIdx.x; j < division; j += blockDim.x) {
      float window_left = width * i + RANGE_MIN;
      o1 += simpson_f12_multi(window_left, width, x);
      o2 += simpson_f12_multi(window_left, width, y);
      o3 += simpson_f3_multi(window_left, width, x, y);
    }

    sdata1[threadIdx.x] = o1;
    sdata2[threadIdx.x] = o2;
    sdata3[threadIdx.x] = o3;
    reduceBlock<TPB>(sdata1, sdata2, sdata3);

    __syncthreads();

    if (threadIdx.x == 0) {
      float int1 = sdata1[0];
      float int2 = sdata2[0];
      float int3 = sdata3[0];
      if (int1 < VERYSMALL) int1 = VERYSMALL;
      if (int2 < VERYSMALL) int2 = VERYSMALL;
      if (int3 < 0.0f)      int3 = 0.0f;

      float dust = -log10(int3 / (int1 * int2));
      if (dust < 0.0) { dust = 0.0f; }

      dusts[i] = dust;
    }
  }
}
