#pragma once

//#undef _GLIBCXX_USE_C99_MATH

#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "randomvariable.hpp"
#include "integrator.hpp"

#include <map>


#define STDDEV_STEP 0.2
#define STDDEV_BEGIN 0.0
#define STDDEV_END 2.0
#define STDDEV_STEPS (int)(((STDDEV_END-STDDEV_BEGIN)/STDDEV_STEP)+1)

#define DISTANCE_STEP 0.1
#define DISTANCE_BEGIN 0.0
#define DISTANCE_END 10.0
#define DISTANCE_STEPS ((int)((DISTANCE_END-DISTANCE_BEGIN)/DISTANCE_STEP)+1)


class GDUST
{
public:
    ~GDUST();
    GDUST(TimeSeriesCollection &collection, const char *lookUpTablesPath = NULL);

    bool  lookupTablesAvailable;
    float lookuptables[ 3 ][ STDDEV_STEPS + 1 ][ STDDEV_STEPS + 1 ][ DISTANCE_STEPS + 1 ];

    float  distance (TimeSeries &ts1, TimeSeries &ts2, int n = -1);
    double dtw (TimeSeries &ts1, TimeSeries &ts2);
    
    void init();

    TimeSeriesCollection collection;

private:
    Integrator *integ;
};

