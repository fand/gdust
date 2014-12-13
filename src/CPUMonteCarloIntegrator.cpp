#include "CPUIntegrator.hpp"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include "RandomVariable.hpp"
#include "config.hpp"
#include "ckernel.hpp"


CPUMonteCarloIntegrator::CPUMonteCarloIntegrator() {
  this->T = gsl_rng_default;
  this->r_rng = gsl_rng_alloc(this->T);
}

CPUMonteCarloIntegrator::~CPUMonteCarloIntegrator() {
  gsl_rng_free(r_rng);
}


//!
// Compute DUST for 2 time series.
//
double
CPUMonteCarloIntegrator::distance(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
  double dist = 0.0;

  const int64_t seeds[12] = {
    3467, 10267, 16651, 19441, 23497, 27361,
    35317, 76801, 199933, 919393, 939193, 999331,
  };

  const int num_rands = n * 3 * INTEGRATION_SAMPLES;
  __attribute__((aligned(64))) double *rands = new double[num_rands];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    struct drand48_data buffer;
    srand48_r(seeds[tid], &buffer);
#pragma omp for schedule(static)
    for (int i = 0; i < num_rands; ++i) {
      drand48_r(&buffer, &rands[i]);
      rands[i] = rands[i] * RANGE_WIDTH + RANGE_MIN;
    }

#pragma omp for reduction(+: dist)
    for (int i = 0; i < n; ++i) {
      RandomVariable x = ts1.at(i);
      RandomVariable y = ts2.at(i);

      double params[] = {
        x.distribution, x.observation, x.stddev,
        y.distribution, y.observation, y.stddev
      };

      double d = c_dust_kernel(params, rands, i);
      dist += d;
    }
  }

  delete[] rands;

  return sqrt(dist);
}
