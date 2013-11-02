#pragma once

#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "randomvariable.hpp"

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

#include <map>

#define INTEGRATION_SAMPLES 49152

//50000

#define STDDEV_STEP 0.2
#define STDDEV_BEGIN 0.0
#define STDDEV_END 2.0
#define STDDEV_STEPS (int)(((STDDEV_END-STDDEV_BEGIN)/STDDEV_STEP)+1)

#define DISTANCE_STEP 0.1
#define DISTANCE_BEGIN 0.0
#define DISTANCE_END 10.0
#define DISTANCE_STEPS ((int)((DISTANCE_END-DISTANCE_BEGIN)/DISTANCE_STEP)+1)


struct tuple_t {
    double x;
    double y;
    double m;
};


class DUST
{
public:
    ~DUST();
    DUST( TimeSeriesCollection &collection, const char *lookUpTablesPath = NULL );
    std::vector< int > rangeQuery( TimeSeries ts, double threshold );

    bool lookupTablesAvailable;
    double lookuptables[ 3 ][ STDDEV_STEPS + 1 ][ STDDEV_STEPS + 1 ][ DISTANCE_STEPS + 1 ];

    double distance( TimeSeries &ts1, TimeSeries &ts2, int n = -1 );
    double dtw( TimeSeries &ts1, TimeSeries &ts2 );

    double integrate( double (*f)( double * x_array, size_t dim, void * params), void *params );

    void init();
    void readLookUpTables( const char *lookUpTablesPath );

    double dust( RandomVariable &x, RandomVariable &y);
    double phi( RandomVariable &x, RandomVariable &y);

    void buildFDustTables( const char *path );

    gsl_rng *r_rng;
    const gsl_rng_type *T;

    TimeSeriesCollection collection;

private:
    void calcCost( TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2 );
    void calcGamma( double *table_d, double *table_g, int len1, int len2 );
    double calcSum( double *table_d, double *table_g, int len1, int len2 );
};
