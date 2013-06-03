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


// struct tuple_t {
//     float x;
//     float y;
//     float m;
// };


class GDUST
{
public:
    ~GDUST();
    GDUST( TimeSeriesCollection &collection, const char *lookUpTablesPath = NULL );
    thrust::host_vector< int > rangeQuery( TimeSeries ts, float threshold );

    bool lookupTablesAvailable;
    float lookuptables[ 3 ][ STDDEV_STEPS + 1 ][ STDDEV_STEPS + 1 ][ DISTANCE_STEPS + 1 ];

    float distance( TimeSeries &ts1, TimeSeries &ts2, int n = -1 );
    double dtw( TimeSeries &ts1, TimeSeries &ts2 );
    
    void init();

    float dust( RandomVariable &x, RandomVariable &y);
    float phi( RandomVariable &x, RandomVariable &y);
    
    void buildFDustTables( const char *path );
    void readLookUpTables( const char *lookUpTablesPath );
    TimeSeriesCollection collection;


private:
    Integrator *integ;
    void calcCost( TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2 );
    void calcGamma( double *table_d, double *table_g, int len1, int len2 );
    double calcSum( double *table_d, double *table_g, int len1, int len2 );
};

