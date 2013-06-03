#pragma once

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>

#define SAMPLES_MAX 32
#define TIMESERIES_MAX 1000

class DataSet
{
public:
    DataSet();
    void normalize( thrust::host_vector< float > &timeSeries );
    void perturbateNothing();
    void perturbateNormal( float mu, float sigma );
    void randomWalks( long n, long length );
    void readFile( const char *src );
    void writeMultiSamplesDir( const char *dst );

    thrust::host_vector< float > originalTimeSeries[ TIMESERIES_MAX ];
    thrust::host_vector< float > perturbatedTimeSeries[ TIMESERIES_MAX ][ SAMPLES_MAX ];
    int N;
};

