#pragma once

#include <curand.h>
#include "timeseries.hpp"


class Integrator
{
public:
    ~Integrator();
    Integrator();    

    float distance(TimeSeries &ts1, TimeSeries &ts2, int n);
    
private:
    curandGenerator_t *gen;
};

