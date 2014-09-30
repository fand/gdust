#pragma once

#include <vector>
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"


class Euclidean {
 public:
  explicit Euclidean(const TimeSeriesCollection &collection, bool exact = false);
  std::vector< int > rangeQuery(const TimeSeries &ts, float threshold);
  float distance(const TimeSeries &ts1, const TimeSeries &ts2, int n = -1);
  double dtw(const TimeSeries &ts1, const TimeSeries &ts2);

  int largestDistanceId;
  TimeSeriesCollection collection;

  float getHeuristicThreshold(float abovePercentual);

  bool exact;

 private:
  void calcCost(const TimeSeries &ts1, const TimeSeries &ts2,
                double *table_d, double *table_g, int len1, int len2);
  void calcGamma(double *table_d, double *table_g, int len1, int len2);
  double calcSum(double *table_d, double *table_g, int len1, int len2);
};
