#pragma once

#include <vector>
#include "Distance.hpp"
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"


class Euclidean : public Distance {
 public:
  explicit Euclidean(const TimeSeriesCollection &collection, bool exact = false);
  ~Euclidean() {};

  double distance(const TimeSeries &ts1, const TimeSeries &ts2, int n = -1);
  float getHeuristicThreshold(float abovePercentual);

  bool exact;  // Use groundtruth or not

 private:
  int largestDistanceId;
  const TimeSeriesCollection *collection;
};
