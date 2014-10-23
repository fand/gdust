#include "Euclidean.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "common.hpp"


Euclidean::Euclidean(const TimeSeriesCollection &collection, bool exact) {
  this->collection = &collection;
  this->exact = exact;
  largestDistanceId = -1;
  std::cout << "this: " << this << std::endl;
  std::cout << "db size: " << this->collection->sequences.size() << std::endl;
}

double Euclidean::distance(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
  int ts_length = std::min(ts1.length(), ts2.length());
  ts_length = (n == -1) ? ts_length : std::min(ts_length, n);

  double dist = 0.0;
  if (exact) {
    for (int i = 0; i < ts_length; i++) {
      dist += pow(ts1.at(i).groundtruth - ts2.at(i).groundtruth, 2);
    }
  } else {
    for (int i = 0; i < ts_length; i++) {
      dist += pow(ts1.at(i).observation -ts2.at(i).observation, 2);
    }
  }

  dist = sqrt(dist);
  return dist;
}

float Euclidean::getHeuristicThreshold(float abovePercentual) {
  float histogram[1000];
  for (int i = 0; i < 1000; i++) {
    histogram[i] = 0;
  }

  float min = -1, max = -1;

  for (unsigned int i = 0; i < collection->sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection->sequences.size(); j++) {
      const TimeSeries &t1 = collection->sequences[i];
      const TimeSeries &t2 = collection->sequences[j];
      float d = distance(t1, t2);

      if (min == -1) {
        min = d;
        max = d;
      } else {
        min = MIN(min, d);
        max = MAX(max, d);
      }
    }
  }

  assert(min <= max && min >= 0);

  float nmax = max - min;

  float alldists = pow(collection->sequences.size(), 2.0) / 2.0 - collection->sequences.size();

  for (unsigned int i = 0; i < collection->sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection->sequences.size(); j++) {
      const TimeSeries &t1 = collection->sequences[i];
      const TimeSeries &t2 = collection->sequences[j];
      float d = distance(t1, t2) - min;
      histogram[static_cast<int>(ceil(d / nmax * (1000 - 1)))]++;
    }
  }

  int64_t counter = 0;
  int bucket_id = 0;
  for (bucket_id = 0; bucket_id < 1000; bucket_id++) {
    counter += histogram[bucket_id];
    if (counter > alldists*abovePercentual) {
      break;
    }
  }

  assert(bucket_id < 1000);

  float dd = min + static_cast<float>(bucket_id) / (1000.0 - 1.0) * nmax;

  counter = 0;

  for (unsigned int i = 0; i < collection->sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection->sequences.size(); j++) {
      const TimeSeries &t1 = collection->sequences[i];
      const TimeSeries &t2 = collection->sequences[j];
      if (distance(t1, t2) <= dd) {
        counter++;
      }
    }
  }

  float coverage = static_cast<float>(counter) / alldists;
  assert(coverage - 0.1 < abovePercentual && coverage + 0.1 > abovePercentual);
  return dd;
}
