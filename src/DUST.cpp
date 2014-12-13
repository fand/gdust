#include "DUST.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include "common.hpp"
#include "config.hpp"
#include "ckernel.hpp"

DUST::DUST(const TimeSeriesCollection &collection, const CPUIntegrator::Method method) {
  this->collection = &collection;
  this->integrator = CPUIntegrator::create(method);
}

DUST::~DUST() {}

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
  return this->integrator->distance(ts1, ts2, lim);
}

int
DUST::match(const TimeSeries &ts) {
  const TimeSeriesCollection *db = this->collection;

  int ts_length = ts.length();
  for (int i = 0; i < db->sequences.size(); i++) {
    ts_length = fmin(ts_length, db->sequences[i].length());
  }

  float distance_min = this->integrator->distance(ts, db->sequences[0], ts_length);
  float i_min = 0;
  for (int i = 1; i < db->sequences.size(); i++) {
    float d = this->integrator->distance(ts, db->sequences[i], ts_length);
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
