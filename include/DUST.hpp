#pragma once

#include <vector>
#include <map>
#include "Distance.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"
#include "CPUIntegrator.hpp"


class DUST : public Distance {
 public:
  explicit DUST(const TimeSeriesCollection &collection,
                const CPUIntegrator::Method method = CPUIntegrator::MonteCarlo);
  ~DUST();

  double difference(const RandomVariable &r1, const RandomVariable &r2);
  double distance(const TimeSeries &ts1, const TimeSeries &ts2, int n = -1);
  int match(const TimeSeries &ts1);
  double dust(const RandomVariable &x, const RandomVariable &y);

 private:
  CPUIntegrator *integrator;
  double phi(const RandomVariable &x, const RandomVariable &y);
};
