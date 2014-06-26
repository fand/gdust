#pragma once

#include <curand.h>
#include "timeseries.hpp"
#include "timeseriescollection.hpp"


class Integrator
{
public:
    ~Integrator();
    Integrator();

    float distance(TimeSeries &ts1, TimeSeries &ts2, int n);
    void match_naive(TimeSeries &ts, TimeSeriesCollection &db);
    void match(TimeSeries &ts, TimeSeriesCollection &db);

private:
    curandGenerator_t *gen;
};
