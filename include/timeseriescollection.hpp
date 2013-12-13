#pragma once

#include "timeseries.hpp"
#include <vector>

typedef std::pair< int, float > PairIdDistance;


class TimeSeriesCollection
{
public:
    TimeSeriesCollection();
    TimeSeriesCollection(const char *pathname, int distribution, int limitN = -1);

    void readFile(const char *pathname, int distribution, int limitN = -1);

    void computeTopKthresholds( unsigned int K, std::vector< PairIdDistance > &topIdDist);

    std::vector< TimeSeries > sequences;

    void normalize();
    void printSeqs();
};

