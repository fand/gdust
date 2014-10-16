#pragma once

#include <vector>
#include <map>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include "Distance.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"


class DUST : public Distance {
 public:
  explicit DUST(const TimeSeriesCollection &collection);
  ~DUST();

  double difference(const RandomVariable &r1, const RandomVariable &r2);
  double distance(const TimeSeries &ts1, const TimeSeries &ts2, int n = -1);
  int match(const TimeSeries &ts1);
  double dust(const RandomVariable &x, const RandomVariable &y);

 private:
  double distance_inner(const TimeSeries &ts1, const TimeSeries &ts2, int n);
  double phi(const RandomVariable &x, const RandomVariable &y);

  TimeSeriesCollection collection;
  gsl_rng *r_rng;
  const gsl_rng_type *T;
};
