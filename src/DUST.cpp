#include "DUST.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>
#include "common.hpp"
#include "config.hpp"
#include "ckernel.hpp"


inline double
clean_probability(double p) {
  return (p < 0.0) ? 0.0 : p;
}


DUST::DUST(const TimeSeriesCollection &collection) {
  this->collection = &collection;
  this->T = gsl_rng_default;
  this->r_rng = gsl_rng_alloc(this->T);
}

DUST::~DUST() {
  gsl_rng_free(r_rng);
}

double
DUST::distance_inner(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
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

double
DUST::phi(const RandomVariable &x, const RandomVariable &y) {
  return 0;
}

double
DUST::distance(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
  int lim;
  if (n == -1) {
    lim = fmin(ts1.length(), ts2.length());
  } else {
    lim = fmin(n, fmin(ts1.length(), ts2.length()));
  }
  return this->distance_inner(ts1, ts2, lim);
}

int
DUST::match(const TimeSeries &ts) {
  const TimeSeriesCollection *db = this->collection;

  int ts_length = ts.length();
  for (int i = 0; i < db->sequences.size(); i++) {
    ts_length = fmin(ts_length, db->sequences[i].length());
  }

  float distance_min = this->distance_inner(ts, db->sequences[0], ts_length);
  float i_min = 0;
  for (int i = 1; i < db->sequences.size(); i++) {
    float d = this->distance_inner(ts, db->sequences[i], ts_length);
    if (d < distance_min) {
      distance_min = d;
      i_min = i;
    }
  }

  std::cout << "matched : " << ts_length << std::endl;
  std::cout << "\t index: " << i_min
            << ", distance : " << distance_min << std::endl;
  return i_min;
}

double
DUST::dust(const RandomVariable &x, const RandomVariable &y) {
  // K disabled. we do not need normalization.
  const double K = 0.0;

  double distance = -log10(phi(x, y)) - K;
  if (distance < 0) {
    distance = 0;
  }
  distance = sqrt(distance);
  assert(NOTNANINF(distance));

  return distance;
}

double
DUST::difference(const RandomVariable &x, const RandomVariable &y) {
  dust(x, y);
}
