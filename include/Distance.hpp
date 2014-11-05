#pragma once

#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"

class Distance {
 public:
  Distance() {}
  virtual ~Distance() {}

  virtual double difference(const RandomVariable &r1, const RandomVariable &r2);
  virtual double distance(const TimeSeries &ts1, const TimeSeries &ts2, int n = -1) = 0;
  virtual double DTW(const TimeSeries &ts1, const TimeSeries &ts2);
  virtual int match(const TimeSeries &ts);
  virtual std::vector<int> rangeQuery(TimeSeries ts, double threshold);
  virtual std::vector<int> topK(const TimeSeries &ts, int k);

 protected:
  const TimeSeriesCollection *collection;
  void calcCost(const TimeSeries &ts1, const TimeSeries &ts2,
                double *table_d, double *table_g, int len1, int len2);
  void calcGamma(double *table_d, double *table_g, int len1, int len2);
  double calcSum(double *table_d, double *table_g, int len1, int len2);
};
