#include "gdust.hpp"
#include "integrator.hpp"
#include "common.hpp"

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>

#define VERYSMALL 1E-20


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


GDUST::GDUST (TimeSeriesCollection &collection, const char *lookUpTablesPath)
{
    this->collection = collection;    
    this->init();

    this->integ = new Integrator();
}


float
GDUST::distance (TimeSeries &ts1, TimeSeries &ts2, int n)
{
    int lim;
    if (n == -1) {
        lim = min(ts1.length(), ts2.length());
    }
    else {
        lim = min(n, min(ts1.length(), ts2.length()));
    }
    return (this->integ)->distance(ts1, ts2, lim);
}

