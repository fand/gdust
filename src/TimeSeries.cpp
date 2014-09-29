#include "TimeSeries.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "RandomVariable.hpp"


TimeSeries::TimeSeries() {
  this->id = -1;
}

TimeSeries::TimeSeries(std::vector<RandomVariable> seq) {
  this->id = -1;
  this->sequence = seq;
}

TimeSeries::TimeSeries(const char *path, int distribution) {
  this->readFile(path, distribution);
}

TimeSeries::TimeSeries(const std::string &s, int distribution) {
  this->readString(s, distribution);
}

void
TimeSeries::readString(const std::string &s, int distribution) {
  // clear sequence data before reading string
  this->sequence.clear();

  std::stringstream lineStream(s);
  std::string str_line;
  float stddev, observation, groundtruth;
  while (lineStream >> str_line) {
    std::replace(str_line.begin(), str_line.end(), ':', ' ');
    std::stringstream line(str_line);

    line >> groundtruth;
    line >> observation;
    line >> stddev;

    RandomVariable r = new RandomVariable(distribution, groundtruth, observation, stddev);
    this->sequence.push_back(r);
  }
}

void
TimeSeries::readFile(const char *path, int distribution) {
  std::ifstream fin(path);
  std::string line;

  if (!getline(fin, line)) {
    std::cout << "Error: cannot open file(" << path << ")" << std::endl;
  }

  this->readString(line, distribution);

  std::cout << "TimeSeries " << path << " : length " << this->sequence.size() << std::endl;
  fin.close();
}

RandomVariable TimeSeries::at(int index) {
  return this->sequence[index];
}

unsigned int
TimeSeries::length() {
  return this->sequence.size();
}

int
TimeSeries::getId() {
  return this->id;
}

void
TimeSeries::setId(int newId) {
  this->id = newId;
}

void
TimeSeries::normalize() {
  float min = this->sequence[0].observation;
  float max = this->sequence[0].observation;
  float tmp = 0.0;

  for (unsigned int i = 1; i < this->sequence.size(); i++) {
    tmp = this->sequence[i].observation;
    if (min > tmp) min = tmp;
    if (max < tmp) max = tmp;
  }

  if (min == max) return;

  float ratio = 1.0f / abs(max-min);

  for (unsigned int i = 0; i < this->sequence.size(); i++) {
    this->sequence[i].groundtruth = this->sequence[i].groundtruth * ratio;
    this->sequence[i].observation = this->sequence[i].observation * ratio;
    this->sequence[i].stddev = this->sequence[i].stddev * ratio;
  }
}

void
TimeSeries::printSeq() {
  for (unsigned int i = 1; i < this->sequence.size(); i++) {
    std::cout << this->sequence[i].observation << std::endl;
  }
}
