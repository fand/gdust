#pragma once

#include "timeseries.hpp"
#include <thrust/host_vector.h>

typedef std::pair< int, float > PairIdDistance;


class TimeSeriesCollection
{
public:
    TimeSeriesCollection();
    TimeSeriesCollection( const char *pathname, int distribution, int limitN = -1 );

    void readFile( const char *pathname, int distribution, int limitN = -1 );

    void computeTopKthresholds( unsigned int K, thrust::host_vector< PairIdDistance > &topIdDist );

    thrust::host_vector< TimeSeries > sequences;

    void normalize();
    void printSeqs();
};

