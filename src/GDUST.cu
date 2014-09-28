// Class for DUST with GPU.
// Actual computation is done by Integrator,
// because it's difficult to separate from GPU management.

#include "GDUST.hpp"
#include "Integrator.hpp"
#include "common.hpp"

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>


inline float
clean_probability (float p)
{
    if(p <= 0) p = 0;
    return p;
}


void
GDUST::init()
{
    lookupTablesAvailable = false;
}


GDUST::~GDUST(){}


GDUST::GDUST (TimeSeriesCollection &collection, const Integrator::Method method)
{
    this->collection = collection;
    this->integ = Integrator::create(method);
    this->init();
}

float
GDUST::distance (TimeSeries &ts1, TimeSeries &ts2, int n)
{
    int ts_length = min(ts1.length(), ts2.length());
    ts_length = (n == -1) ? ts_length : min(ts_length, n);
    return (this->integ)->distance(ts1, ts2, ts_length);
}

void
GDUST::match_naive (TimeSeries &ts)
{
    this->integ->match_naive(ts, this->collection);
}

void
GDUST::match (TimeSeries &ts)
{
    this->integ->match(ts, this->collection);
}
