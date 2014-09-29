#pragma once

#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>


#define SAMPLES_MAX 32
#define TIMESERIES_MAX 1000

class DataSet {
 public:
  DataSet();
  void normalize(const std::vector< float > &timeSeries);
  void perturbateNothing();
  void perturbateNormal(float mu, float sigma);
  void randomWalks(int64 n, int64 length);
  void readFile(const char *src);
  void writeMultiSamplesDir(const char *dst);

  std::vector< float > originalTimeSeries[TIMESERIES_MAX];
  std::vector< float > perturbatedTimeSeries[TIMESERIES_MAX][SAMPLES_MAX];
  int N;
};
