#pragma once

#include "TimeSeries.hpp"
#include "TimeSeriesCollection.hpp"
#include "RandomVariable.hpp"

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

#include <map>

#define STDDEV_STEP    0.2
#define STDDEV_BEGIN   0.0
#define STDDEV_END     2.0
#define STDDEV_STEPS   ((int)(((STDDEV_END - STDDEV_BEGIN)/STDDEV_STEP) + 1))

#define DISTANCE_STEP  0.1
#define DISTANCE_BEGIN 0.0
#define DISTANCE_END   10.0
#define DISTANCE_STEPS ((int)((DISTANCE_END - DISTANCE_BEGIN)/DISTANCE_STEP) + 1)



class DUST
{
public:
    DUST(const TimeSeriesCollection &collection, const char *lookUpTablesPath = NULL);
    ~DUST();

    void init();

    TimeSeriesCollection collection;

    double integrate (double (*f)(double *x_array, size_t dim, void *params), void *params);

    double dust (const RandomVariable &x, const RandomVariable &y);
    double phi (const RandomVariable &x, const RandomVariable &y);

    double distance (const TimeSeries &ts1, const TimeSeries &ts2, int n = -1);
    double c_distance (const TimeSeries &ts1, const TimeSeries &ts2, int n);
    double dtw (const TimeSeries &ts1, const TimeSeries &ts2);

    void match (const TimeSeries &ts, int n = -1);

    std::vector<int> rangeQuery (TimeSeries ts, double threshold);

    void   buildFDustTables (const char *path);
    void   readLookUpTables (const char *lookUpTablesPath);

    gsl_rng *r_rng;
    const gsl_rng_type *T;
    bool   lookupTablesAvailable;
    double lookuptables[ 3 ][ STDDEV_STEPS + 1 ][ STDDEV_STEPS + 1 ][ DISTANCE_STEPS + 1 ];


private:
    void calcCost  (TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2);
    void calcGamma (double *table_d, double *table_g, int len1, int len2);
    double calcSum (double *table_d, double *table_g, int len1, int len2);
};


// Global functions for integration.
double f1 (double *k, size_t dim, void *params);
double f2 (double *k, size_t dim, void *params);
double f3 (double *k, size_t dim, void *params);
