// Class for DUST with GPU.
// Actual computation is done by Integrator,
// because it's difficult to separate from GPU management.

#include "GDUST.hpp"
#include <algorithm>
#include "Integrator.hpp"
#include "common.hpp"


inline float
clean_probability(float p) {
  if (p <= 0) p = 0;
  return p;
}

GDUST::GDUST(const TimeSeriesCollection &collection, const Integrator::Method method) {
  collection = collection;
  integrator = Integrator::create(method);
}

GDUST::~GDUST() {}

float
GDUST::distance(const TimeSeries &ts1, const TimeSeries &ts2, const int n) {
  int ts_length = min(ts1.length(), ts2.length());
  ts_length = (n == -1) ? ts_length : min(ts_length, n);
  return integrator->distance(ts1, ts2, ts_length);
}

void
GDUST::match_naive(const TimeSeries &ts) {
  this->integrator->match_naive(ts, this->collection);
}

void
GDUST::match(const TimeSeries &ts) {
  integrator->match(ts, this->collection);
}
