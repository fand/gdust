#pragma once

//#undef _GLIBCXX_USE_C99_MATH

#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "randomvariable.hpp"
#include "integrator.hpp"

#include <map>



class GDUST
{
public:
    ~GDUST();
    GDUST(TimeSeriesCollection &collection, const char *lookUpTablesPath = NULL);

    void init();
    
    float  distance (TimeSeries &ts1, TimeSeries &ts2, int n = -1);
    double dtw (TimeSeries &ts1, TimeSeries &ts2);
    
    TimeSeriesCollection collection;

    bool  lookupTablesAvailable;


private:
    Integrator *integ;
};

