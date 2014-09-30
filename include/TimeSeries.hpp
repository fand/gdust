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
  RandomVariable operator [] (int index) const;
  RandomVariable at(int index) const;
  float setValueAt(int index, float value);
  RandomVariable setValueAt(int index, RandomVariable value);

  unsigned int length() const;
  int getId() const;
  void setId(int newId);
  void normalize();
  void printSeq() const;

 private:
  std::vector< RandomVariable > sequence;
  int id;
};
