#pragma once

#include <map>
#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"
#include "Integrator.hpp"


class GDUST {
public:
GDUST(const TimeSeriesCollection &collection, const Integrator::Method method = Integrator::MonteCarlo);
~GDUST();

void init();

float distance(const TimeSeries &ts1, const TimeSeries &ts2, const int n = -1);
double dtw (const TimeSeries &ts1, const TimeSeries &ts2);

void match_naive (const TimeSeries &ts);
void match (const TimeSeries &ts);

TimeSeriesCollection collection;
bool  lookupTablesAvailable;

private:
Integrator *integ;
};
