#include "CPUIntegrator.hpp"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include "RandomVariable.hpp"
#include "config.hpp"
#include "ckernel.hpp"

#define DIVISION 32

CPUSimpsonIntegrator::CPUSimpsonIntegrator() {
}

CPUSimpsonIntegrator::~CPUSimpsonIntegrator() {

}


//!
// Compute DUST for 2 time series.
//
double
CPUSimpsonIntegrator::distance(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
  double dist = 0.0;

  for (int i = 0; i < n; ++i) {
    RandomVariable x = ts1.at(i);
    RandomVariable y = ts2.at(i);

    double params[] = {
      x.distribution, x.observation, x.stddev,
      y.distribution, y.observation, y.stddev
    };

    double d = this->dust_kernel(params, i);
    dist += d;
  }

  return sqrt(dist);
}

double
CPUSimpsonIntegrator::dust_kernel(double *xy, int time) {
  int division_all = DIVISION * RANGE_WIDTH;
  double width = 1.0f / static_cast<double>(DIVISION);

  double int1, int2, int3;
  int1 = int2 = int3 = 0.0;

  for (int i = 0; i < division_all; i++) {
    double window_left = width * i + RANGE_MIN;
    int1 += this->f1(window_left, width, xy);
    int2 += this->f2(window_left, width, xy);
    int3 += this->f3(window_left, width, xy);
  }

  if (int1 < VERYSMALL) int1 = VERYSMALL;
  if (int2 < VERYSMALL) int2 = VERYSMALL;
  if (int3 < 0.0f)      int3 = 0.0f;

  double dust = -log10(int3 / (int1 * int2));
  if (dust < 0.0) { dust = 0.0f; }

  return  dust;
}

double
CPUSimpsonIntegrator::f1(double left, double width, double *tuple) {
  double mid = left + width * 0.5;
  double right = left + width;
  return (width / 6) * (c_f1(left, tuple) +
                        c_f1(right, tuple) +
                        c_f1(mid, tuple) * 4);
}
double
CPUSimpsonIntegrator::f2(double left, double width, double *tuple) {
  double mid = left + width * 0.5;
  double right = left + width;
  return (width / 6) * (c_f2(left, tuple) +
                        c_f2(right, tuple) +
                        c_f2(mid, tuple) * 4);
}
double
CPUSimpsonIntegrator::f3(double left, double width, double *tuple) {
  double mid = left + width * 0.5;
  double right = left + width;
  return (width / 6) * (c_f3(left, tuple) +
                        c_f3(right, tuple) +
                        c_f3(mid, tuple) * 4);
}
