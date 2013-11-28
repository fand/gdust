#pragma once

#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include <vector>

class Euclidean
{
public:
    Euclidean( TimeSeriesCollection &collection, bool exact = false );
    std::vector< int > rangeQuery( TimeSeries &ts, float threshold );
    float distance( TimeSeries &ts1, TimeSeries &ts2, int n = -1 );
    double dtw( TimeSeries &ts1, TimeSeries &ts2 );
    
    int largestDistanceId;
    TimeSeriesCollection collection;

    float getHeuristicThreshold( float abovePercentual );

    bool exact;

private:
    void calcCost( TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2 );
    void calcGamma( double *table_d, double *table_g, int len1, int len2 );
    double calcSum( double *table_d, double *table_g, int len1, int len2 );
};
