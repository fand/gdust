#pragma once

#include <string>
#include <vector>
#include "RandomVariable.hpp"


class TimeSeries {
 public:
  TimeSeries();
  explicit TimeSeries(std::vector< RandomVariable > seq);
  TimeSeries(const char *path, int distribution);
  TimeSeries(const std::string &s, int distribution);
  void readFile(const char *path, int distribution);
  void readString(const std::string &s, int distribution);
  RandomVariable at(int index);

  unsigned int length();
  int getId();
  void setId(int newId);
  void normalize();
  void printSeq();

 private:
  std::vector< RandomVariable > sequence;
  int id;
};
