#pragma once

#include <map>
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"
#include "Integrator.hpp"


class GDUST {
 public:
  explicit GDUST(const TimeSeriesCollection &collection,
                 const Integrator::Method method = Integrator::MonteCarlo);
  ~GDUST();

  float distance(const TimeSeries &ts1, const TimeSeries &ts2, const int n = -1);
  void match_naive(const TimeSeries &ts);
  void match(const TimeSeries &ts);

 private:
  TimeSeriesCollection collection;
  Integrator *integrator;
};
