#pragma once
#include <curand.h>
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"

///
// Integrators for dust
//

class Integrator {
 public:
  enum Method { MonteCarlo, Simpson };
  static Integrator* create(Method method);

  Integrator() {}
  virtual ~Integrator() {}

  virtual float distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length) = 0;
  virtual int match_naive(const TimeSeries &ts, const TimeSeriesCollection &db) = 0;
  virtual int match(const TimeSeries &ts, const TimeSeriesCollection &db) = 0;

 protected:
  float *ts_H, *tsc_H, *dusts_H;
  float *ts_D, *tsc_D, *dusts_D;
  size_t ts_size, tsc_size, dusts_size;
  int ts_num, ts_length;
  void prepare_match(const TimeSeries &ts, const TimeSeriesCollection &tsc);
  void finish_match(int *i_min, float *DUST_min);
};

class MonteCarloIntegrator : public Integrator {
 public:
  MonteCarloIntegrator();
  ~MonteCarloIntegrator();
  float distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length);
  int match_naive(const TimeSeries &ts, const TimeSeriesCollection &db);
  int match(const TimeSeries &ts, const TimeSeriesCollection &db);

 private:
  curandGenerator_t *gen;
  float *samples_H, *samples_D;
  size_t samples_size;
};

class SimpsonIntegrator : public Integrator {
 public:
  SimpsonIntegrator();
  ~SimpsonIntegrator();
  float distance(const TimeSeries &ts1, const TimeSeries &ts2, int ts_length);
  int match_naive(const TimeSeries &ts, const TimeSeriesCollection &db);
  int match(const TimeSeries &ts, const TimeSeriesCollection &db);
};
