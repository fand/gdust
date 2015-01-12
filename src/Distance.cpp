#include "Distance.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

std::vector<int>
Distance::rangeQuery(const TimeSeries ts, double threshold) {
  std::vector< int > ids;

  for (int i = 0; i < collection->size(); i++) {
    double dist = this->distance(ts, collection->at(i));
    if (dist <= threshold) {
      ids.push_back(i);
    }
  }

  return ids;
}

double
Distance::DTW(const TimeSeries &ts1, const TimeSeries &ts2) {
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

void
Distance::calcCost(const TimeSeries &ts1, const TimeSeries &ts2,
                   double *table_d, double *table_g, int len1, int len2) {
  for (int i = 0; i < len1; i++) {
    RandomVariable r1 = ts1.at(i);

    for (int j = 0; j < len2; j++) {
      RandomVariable r2 = ts2.at(j);
      table_d[i * len2 + j] = difference(r1, r2);
    }
  }
}

void
Distance::calcGamma(double *table_d, double *table_g, int len1, int len2) {
  table_g[0] = table_d[0];

  for (int i = 1; i < len1; i++) {
    table_g[i * len2] = table_d[i * len2] + table_g[(i - 1) * len2];
  }

  for (int i = 1; i < len2; i++) {
    table_g[i] = table_d[i] + table_g[i - 1];
  }

  for (int i = 1; i< len1; i++) {
    for (int j = 1; j < len2; j++) {
      table_g[i * len2 + j] = table_d[i * len2 + j] +
        std::min(
                 std::min(table_g[(i - 1) * len2 + j],
                          table_g[i * len2 + j - 1]),
                 table_g[(i - 1) * len2 + j - 1]);
    }
  }
}

double
Distance::calcSum(double *table_d, double *table_g, int len1, int len2) {
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
      double m = std::min(
                          std::min(table_g[(i - 1) * len2 + j],
                                   table_g[i * len2 + j-1]),
                          table_g[(i - 1) * len2 + j - 1]);

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

double
Distance::difference(const RandomVariable &r1, const RandomVariable &r2) {
  return abs(r1.observation - r2.observation);
}

int
Distance::match(const TimeSeries &ts) {
  const TimeSeriesCollection *db = this->collection;

  int ts_length = ts.length();
  for (int i = 0; i < db->sequences.size(); i++) {
    ts_length = std::min(ts_length, (int)db->sequences[i].length());
  }
  float distance_min = this->distance(ts, db->sequences[0], ts_length);
  float i_min = 0;

  for (int i = 1; i < db->sequences.size(); i++) {
    float d = this->distance(ts, db->sequences[i], ts_length);
    if (d < distance_min) {
      distance_min = d;
      i_min = i;
    }
  }

  std::cout << "matched : " << ts_length << std::endl;
  std::cout << "\t index: " << i_min
            << ", distance : " << distance_min << std::endl;
  return i_min;
}


static bool comp(const std::pair<int, float> &l, const std::pair<int, float> &r){
  return l.second < r.second;
}

std::vector<int>
Distance::topK(const TimeSeries &ts, int k) {
  const TimeSeriesCollection *db = this->collection;
  int ts_length = ts.length();
  for (int i = 0; i < db->sequences.size(); i++) {
    ts_length = std::min(ts_length, (int)db->sequences[i].length());
  }

  // Get distances
  std::vector<std::pair<int, float> > pairs;
  for (int i = 0; i < db->sequences.size(); i++) {
    float d = this->distance(ts, db->sequences[i], ts_length);
    pairs.push_back(std::make_pair(i, d));
  }

  // Sort by value
  std::sort(pairs.begin(), pairs.end(), comp);

  // Return the indexes of top K.
  std::vector<int> ret;
  for (int i = 0; i < k; i++) {
    ret.push_back(pairs[i].first);
  }
  return ret;
}

std::pair<int, float>
Distance::topKThreshold(const TimeSeries &ts, int k) {
  const TimeSeriesCollection *db = this->collection;
  int ts_length = ts.length();
  for (int i = 0; i < db->sequences.size(); i++) {
    ts_length = std::min(ts_length, (int)db->sequences[i].length());
  }

  // Get distances
  std::vector<std::pair<int, float> > pairs;
  for (int i = 0; i < db->sequences.size(); i++) {
    float d = this->distance(ts, db->sequences[i], ts_length);
    pairs.push_back(std::make_pair(i, d));
  }

  // Sort by value
  std::sort(pairs.begin(), pairs.end(), comp);

  // Return the indexes of top K.
  return pairs[k - 1];
}
