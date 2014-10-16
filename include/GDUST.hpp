#pragma once

#include <map>
#include "Distance.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"
#include "Integrator.hpp"


class GDUST : public Distance {
 public:
  explicit GDUST(const TimeSeriesCollection &collection,
                 const Integrator::Method method = Integrator::MonteCarlo);
  ~GDUST();

  double distance(const TimeSeries &ts1, const TimeSeries &ts2, const int n = -1);
  int match_naive(const TimeSeries &ts);
  int match(const TimeSeries &ts);

 private:
  const TimeSeriesCollection *collection;
  Integrator *integrator;
};
