#pragma once

#include <iostream>
#include <sys/time.h>
#include <thrust/host_vector.h>

class PrecisionRecallM
{
public:
    PrecisionRecallM( const char *s );

    void addStartTime();
    void addStopTime();
    void add( thrust::host_vector< int > &exact, thrust::host_vector< int > &estimate );
    float getPrecision();
    float getRecall();
    float getF1();
    float getTime();

    void print();

    float n;
    float precision;
    float recall;
    float size;
    float t;
    float tn;
    const char *s;

    struct timeval begin;
};


