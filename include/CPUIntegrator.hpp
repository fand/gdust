#pragma once
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "Distance.hpp"
#include "RandomVariable.hpp"
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>


class CPUIntegrator {
 public:
  enum Method { MonteCarlo, Simpson };
  static CPUIntegrator* create(Method method);

  CPUIntegrator() {}
  virtual ~CPUIntegrator() {}
  virtual double distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length) = 0;
};

class CPUMonteCarloIntegrator : public CPUIntegrator {
 public:
  CPUMonteCarloIntegrator();
  ~CPUMonteCarloIntegrator();
  double distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length);

 private:
  gsl_rng *r_rng;
  const gsl_rng_type *T;
};

class CPUSimpsonIntegrator : public CPUIntegrator {
 public:
  CPUSimpsonIntegrator();
  ~CPUSimpsonIntegrator();
  double distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length);

 private:
  double dust_kernel(double *xy, int time);
  double f1(double left, double width, double *tuple);
  double f2(double left, double width, double *tuple);
  double f3(double left, double width, double *tuple);
};
