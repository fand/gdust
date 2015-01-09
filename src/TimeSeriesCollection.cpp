#include "TimeSeriesCollection.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "Euclidean.hpp"
#include "common.hpp"


TimeSeriesCollection::TimeSeriesCollection() {}

TimeSeriesCollection::TimeSeriesCollection(const char *path, int distribution, int limitN) {
  this->readFile(path, distribution, limitN);
}

void
TimeSeriesCollection::readFile(const char *path, int distribution, int limitN) {
  std::ifstream fin(path);

  if (fin == NULL) {
    FATAL("unable to read timeseriescollection from file");
  }

  std::string line;
  while (getline(fin, line)) {
    if (limitN != -1 && static_cast<int>(sequences.size()) == limitN) {
      break;
    }
    sequences.push_back(TimeSeries(line, distribution));
    sequences[sequences.size() - 1].setId(sequences.size());
  }

  std::cerr << "Read " << sequences.size()
            << " timeseries from " << path
            << ", sequenceslength =  " << sequences.size() << std::endl;

  fin.close();
}

struct mycompare {
  bool operator()(PairIdDistance const &a, PairIdDistance const &b) {
    return a.second < b.second;
  }
};

void
TimeSeriesCollection::computeTopKthresholds(unsigned int K,
                                            std::vector< PairIdDistance > *topIdDists) {
  assert(sequences.size() >= K);
  topIdDists->resize(sequences.size());

  std::cout << "resized" << std::endl;

  Euclidean trueEuclidean(*this, true);

  for (unsigned int i = 0; i < sequences.size(); i++) {
    TimeSeries &q = sequences[i];
    std::vector< PairIdDistance > distances;

    for (unsigned int j = 0; j < sequences.size(); j++) {
      TimeSeries &t = sequences[j];
      PairIdDistance d = PairIdDistance(t.getId(), trueEuclidean.distance(q, t));
      distances.push_back(d);
    }

    std::sort(distances.begin(), distances.end(), mycompare());
    (*topIdDists)[q.getId() - 1] = distances[K];
  }
}

void
TimeSeriesCollection::normalize() {
  for (int i = 0; i < sequences.size(); i++) {
    sequences[i].normalize();
  }
}

void TimeSeriesCollection::printSeqs() const {
  for (int i = 0; i < sequences.size(); i++) {
    sequences[i].printSeq();
  }
}

int TimeSeriesCollection::length_min() const {
  int l = this->sequences[0].length();
  for (int i = 1; i < this->sequences.size(); i++) {
    l = std::min(l, static_cast<int>(this->sequences[i].length()));
  }
  return l;
}

int TimeSeriesCollection::size() const {
  return this->sequences.size();
}

const TimeSeries& TimeSeriesCollection::at(int index) const {
  return this->sequences.at(index);
}
