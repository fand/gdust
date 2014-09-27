#pragma once

#include <curand.h>
#include "timeseries.hpp"
#include "timeseriescollection.hpp"

class Integrator
{
public:
    enum Method { MonteCarlo, Simpson };
    static Integrator* create(Method method);

    Integrator(){};
    virtual ~Integrator(){};

    virtual float distance(TimeSeries &ts1, TimeSeries &ts2, int n) = 0;
    virtual void match_naive(TimeSeries &ts, TimeSeriesCollection &db) = 0;
    virtual void match(TimeSeries &ts, TimeSeriesCollection &db) = 0;
};

class MonteCarloIntegrator : public Integrator
{
public:
    MonteCarloIntegrator();
    ~MonteCarloIntegrator();
    float distance(TimeSeries &ts1, TimeSeries &ts2, int n);
    void match_naive(TimeSeries &ts, TimeSeriesCollection &db);
    void match(TimeSeries &ts, TimeSeriesCollection &db);

private:
    curandGenerator_t *gen;
};

class SimpsonIntegrator : public Integrator
{
public:
    SimpsonIntegrator();
    ~SimpsonIntegrator();
    float distance(TimeSeries &ts1, TimeSeries &ts2, int n);
    void match_naive(TimeSeries &ts, TimeSeriesCollection &db);
    void match(TimeSeries &ts, TimeSeriesCollection &db);

private:
    curandGenerator_t *gen;
};
