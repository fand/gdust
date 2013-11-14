#pragma once

#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

#define SAMPLES_MAX 32
#define TIMESERIES_MAX 1000

class DataSet
{
public:
    DataSet();
    void normalize( std::vector< float > &timeSeries );
    void perturbateNothing();
    void perturbateNormal( float mu, float sigma );
    void randomWalks( long n, long length );
    void readFile( const char *src );
    void writeMultiSamplesDir( const char *dst );

    std::vector< float > originalTimeSeries[ TIMESERIES_MAX ];
    std::vector< float > perturbatedTimeSeries[ TIMESERIES_MAX ][ SAMPLES_MAX ];
    int N;
};

