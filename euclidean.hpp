#pragma once

#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include <thrust/host_vector.h>

class Euclidean
{
public:
    Euclidean( TimeSeriesCollection &collection, bool exact = false );
    thrust::host_vector< int > rangeQuery( TimeSeries &ts, float threshold );
    float distance( TimeSeries &ts1, TimeSeries &ts2, int n = -1 );

    int largestDistanceId;
    TimeSeriesCollection collection;

    float getHeuristicThreshold( float abovePercentual );

    bool exact;
};
