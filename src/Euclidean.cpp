#include "Euclidean.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "common.hpp"


Euclidean::Euclidean(const TimeSeriesCollection &collection, bool exact) {
  this->collection = collection;
  this->exact = exact;
  largestDistanceId = -1;
}

std::vector< int > Euclidean::rangeQuery(const TimeSeries &ts, float threshold) {
  std::vector< int > ids;
  for (unsigned int i = 0; i < collection.sequences.size(); i++) {
    if (distance(ts, collection.sequences[i]) <= threshold) {
      ids.push_back(i + 1);
    }
  }
  return ids;
}

float Euclidean::distance(const TimeSeries &ts1, const TimeSeries &ts2, int n) {
  if (ts1.length() != ts2.length()) {
    FATAL("ts1 length=" + TO_STR(ts1.length()) +
          " != ts2 length=" + TO_STR(ts2.length()));
  }

  if (n == -1) {
    n = ts1.length();
  }

  float dist = 0;
  if (exact) {
    for (int i = 0; i < n; i++) {
      dist += pow(ts1.at(i).groundtruth - ts2.at(i).groundtruth, 2);
    }
  } else {
    for (int i = 0; i < n; i++) {
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

  for (unsigned int i = 0; i < collection.sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection.sequences.size(); j++) {
      TimeSeries &t1 = collection.sequences[i];
      TimeSeries &t2 = collection.sequences[j];
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

  float alldists = pow(collection.sequences.size(), 2.0) / 2.0 - collection.sequences.size();

  for (unsigned int i = 0; i < collection.sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection.sequences.size(); j++) {
      TimeSeries &t1 = collection.sequences[i];
      TimeSeries &t2 = collection.sequences[j];
      float d = distance(t1, t2) - min;
      histogram[static_cast<int>(ceil(d / nmax * (1000 - 1)))]++;
    }
  }

  unsigned__int64 counter = 0;
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

  for (unsigned int i = 0; i < collection.sequences.size(); i++) {
    for (unsigned int j = i+1; j < collection.sequences.size(); j++) {
      TimeSeries &t1 = collection.sequences[i];
      TimeSeries &t2 = collection.sequences[j];
      if (distance(t1, t2) <= dd) {
        counter++;
      }
    }
  }

  float coverage = static_cast<float>(counter) / alldists;
  assert(coverage - 0.1 < abovePercentual && coverage + 0.1 > abovePercentual);
  return dd;
}

double Euclidean::dtw(const TimeSeries &ts1, const TimeSeries &ts2) {
  int len1 = ts1.length();
  int len2 = ts2.length();
  double *table_d = new double[len1 * len2];
  double *table_g = new double[len1 * len2];

  calcCost(ts1, ts2, table_d, table_g, len1, len2);
  calcGamma(table_d, table_g, len1, len2);
  double dist = calcSum(table_d, table_g, len1, len2);

  delete[] table_d;
  delete[] table_g;
  return dist;
}

void Euclidean::calcCost(const TimeSeries &ts1, const TimeSeries &ts2,
                         double *table_d, double *table_g, int len1, int len2) {
  for (int i = 0; i < len1; i++) {
    RandomVariable r1 = ts1.at(i);
    for (int j = 0; j < len2; j++) {
      RandomVariable r2 = ts2.at(j);
      table_d[i * len2 + j] = abs(r1.observation - r2.observation);
    }
  }
}

void Euclidean::calcGamma(double *table_d, double *table_g, int len1, int len2) {
  table_g[0] = table_d[0];

  for (int i = 1; i < len1; i++) {
    table_g[i * len2] = table_d[i * len2] + table_g[(i - 1) * len2];
  }

  for (int i = 1; i < len2; i++) {
    table_g[i] = table_d[i] + table_g[i - 1];
  }

  for (int i = 1; i< len1; i++) {
    for (int j = 1; j < len2; j++) {
      table_g[i * len2 + j] =
        table_d[i * len2 + j] +
        std::min(std::min(table_g[ (i-1) * len2 + j ],
                          table_g[ i*len2 + j-1 ]),
                 table_g[ (i-1)*len2 + j-1 ]);
    }
  }
}

double Euclidean::calcSum(double *table_d, double *table_g, int len1, int len2) {
  double sum = 0.0;

  int i = len1 - 1;
  int j = len2 - 1;

  while (i > 0 || j > 0) {
    sum += table_g[i * len2 + j];

    if (i == 0) {
      j--;
    } else if (j == 0) {
      i--;
    } else {
      double m = std::min(std::min(table_g[ (i-1)*len2 + j ],
                                   table_g[ i*len2 + j-1 ]),
                          table_g[ (i-1)*len2 + j-1 ]);

      if (m == table_g[(i - 1) * len2 + j]) {
        i--;
      } else if (m == table_g[i * len2 + j - 1]) {
        j--;
      } else {
        i--; j--;
      }
    }
  }

  return sum;
}
