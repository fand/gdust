#include "Integrator.hpp"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <curand.h>
#include "RandomVariable.hpp"
#include "config.hpp"
#include "dust_inner.hpp"
#include "gpu.hpp"
#include "cutil.hpp"


// local functions.
__global__ void
distance_kernel(float *tuples_GPU, float *samples_GPU, float *dust_GPU);
__global__ void
match_kernel(float *ts_GPU, float *tsc_GPU, float *dust_GPU,
             size_t ts_length, size_t ts_num, float *samples);



MonteCarloIntegrator::MonteCarloIntegrator() {
  this->gen = new curandGenerator_t();
  curandCreateGenerator(this->gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(*(this->gen), 1234ULL);
}

MonteCarloIntegrator::~MonteCarloIntegrator() {
  curandDestroyGenerator(*(this->gen));
}


//!
// Compute DUST for 2 time series.
//
float
MonteCarloIntegrator::distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length) {
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

    tuples[idx]     = static_cast<float>(x.distribution);
    tuples[idx + 1] = x.observation;
    tuples[idx + 2] = x.stddev;
    tuples[idx + 3] = static_cast<float>(y.distribution);
    tuples[idx + 4] = y.observation;
    tuples[idx + 5] = y.stddev;
    idx += TUPLE_SIZE;
  }

  checkCudaErrors(cudaMemcpy(tuples_GPU,
                             tuples,
                             tuples_size,
                             cudaMemcpyHostToDevice));

  // generate uniform random number on samples_GPU
  float *samples_GPU;
  checkCudaErrors(cudaMalloc((void**)&samples_GPU,
                             sizeof(float) * INTEGRATION_SAMPLES * ts_length * 3));
  curandGenerateUniform(*(this->gen), samples_GPU, INTEGRATION_SAMPLES * ts_length * 3);


  // call kernel
  distance_kernel<<< ts_length, TPB >>>(tuples_GPU, samples_GPU, dusts_GPU);

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
  checkCudaErrors(cudaFree(samples_GPU));

  return sqrt(dust_sum);
}

// Match 1 ts to all ts in tsc.
// Repeat Integrator::distance for all combination.
void
MonteCarloIntegrator::match_naive(const TimeSeries &ts, const TimeSeriesCollection &tsc) {
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
MonteCarloIntegrator::match(const TimeSeries &ts, const TimeSeriesCollection &tsc) {
  this->prepare_match(ts, tsc);

  // Generate uniform random number on samples_GPU.
  size_t samples_num = INTEGRATION_SAMPLES * ts_length * ts_num * 3;
  checkCudaErrors(cudaMalloc((void**)&(samples_D), sizeof(float) * samples_num));
  curandGenerateUniform(*(this->gen), samples_D, samples_num);

  match_kernel<<< ts_length, TPB >>>(ts_D,
                                tsc_D,
                                dusts_D,
                                ts_length,
                                ts_num,
                                this->samples_D);

  int i_min;
  float DUST_min;
  this->finish_match(&i_min, &DUST_min);
  checkCudaErrors(cudaFree(samples_D));

  std::cout << "matched : " << ts_length << std::endl;
  std::cout << "\t index: " << i_min << ", distance: " << DUST_min << std::endl;
}


//
// Functions for dust / DUST
// ______________________________

// With seq on global memory
__global__ void
distance_kernel(float *tuples_GPU,
                float *samples_GPU,
                float *dust_GPU) {
  float *tuple = tuples_GPU  + blockIdx.x * TUPLE_SIZE;
  float *answer_GPU  = dust_GPU + blockIdx.x;

  float sample1, sample2, sample3;
  int offset1 = blockIdx.x * INTEGRATION_SAMPLES;
  int offset2 = offset1 + INTEGRATION_SAMPLES * gridDim.x;
  int offset3 = offset2 + INTEGRATION_SAMPLES * gridDim.x;

  float o1 = 0.0f;
  float o2 = 0.0f;
  float o3 = 0.0f;

  __shared__ float sdata1[TPB];
  __shared__ float sdata2[TPB];
  __shared__ float sdata3[TPB];

  // MAP PHASE
  // put (f1, f2, f3) into (o1, o2, o3) for all samples
  for (int i = threadIdx.x; i < INTEGRATION_SAMPLES; i += blockDim.x) {
    sample1 = samples_GPU[i + offset1] * RANGE_WIDTH + RANGE_MIN;
    sample2 = samples_GPU[i + offset2] * RANGE_WIDTH + RANGE_MIN;
    sample3 = samples_GPU[i + offset3] * RANGE_WIDTH + RANGE_MIN;
    o1 += f1(sample1, tuple);
    o2 += f2(sample2, tuple);
    o3 += f3(sample3, tuple);
  }

  // REDUCE PHASE
  // Get sum of (o1, o2, o3) for all threads
  sdata1[threadIdx.x] = o1;
  sdata2[threadIdx.x] = o2;
  sdata3[threadIdx.x] = o3;
  reduceBlock<TPB>(sdata1, sdata2, sdata3);

  float r = static_cast<float>(RANGE_WIDTH) / INTEGRATION_SAMPLES;

  if (threadIdx.x == 0) {
    float int1 = sdata1[0] * r;
    float int2 = sdata2[0] * r;
    float int3 = sdata3[0] * r;
    if (int1 < VERYSMALL) int1 = VERYSMALL;
    if (int2 < VERYSMALL) int2 = VERYSMALL;
    if (int3 < 0.0f) int3 = 0.0f;

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
        float *samples) {
  int time = blockIdx.x;

  float sample1, sample2, sample3;
  int offset1 = blockIdx.x * INTEGRATION_SAMPLES;
  int offset2 = offset1 + INTEGRATION_SAMPLES * gridDim.x * ts_num;  // 50000 * lim * 1 + offset1
  int offset3 = offset2 + INTEGRATION_SAMPLES * gridDim.x * ts_num;  // 50000 * lim * 2 + offset1
  float *samples1 = &samples[offset1];
  float *samples2 = &samples[offset2];
  float *samples3 = &samples[offset3];

  float o1 = 0.0f;
  float o2 = 0.0f;
  float o3 = 0.0f;

  __shared__ float sdata1[TPB];
  __shared__ float sdata2[TPB];
  __shared__ float sdata3[TPB];

  float *dusts = &dust_GPU[ts_num * time];
  float r = static_cast<float>(RANGE_WIDTH) / INTEGRATION_SAMPLES;

  float *tsc = &tsc_GPU[ts_num * time * 3];  // db for this block
  float *x  = &ts_GPU[time * 3];             // TODO: compute f1 only once.

  for (int i = 0; i < ts_num; i++) {
    float *y = &tsc[i * 3];
    o1 = o2 = o3 = 0.0f;
    for (int j = threadIdx.x; j < INTEGRATION_SAMPLES; j += blockDim.x) {
      sample1 = samples1[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;
      sample2 = samples2[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;
      sample3 = samples3[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;
      o1 += f12_multi(sample1, x);
      o2 += f12_multi(sample2, y);
      o3 += f3_multi(sample3, x, y);
    }

    sdata1[threadIdx.x] = o1;
    sdata2[threadIdx.x] = o2;
    sdata3[threadIdx.x] = o3;

    reduceBlock<TPB>(sdata1, sdata2, sdata3);

    __syncthreads();

    if (threadIdx.x == 0) {
      float int1 = sdata1[0] * r;
      float int2 = sdata2[0] * r;
      float int3 = sdata3[0] * r;
      if (int1 < VERYSMALL) int1 = VERYSMALL;
      if (int2 < VERYSMALL) int2 = VERYSMALL;
      if (int3 < 0.0f)      int3 = 0.0f;

      float dust = -log10(int3 / (int1 * int2));
      if (dust < 0.0) { dust = 0.0f; }

      dusts[i] = dust;
    }
  }
}
