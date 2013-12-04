#pragma once

#include <iostream>
#include <sys/time.h>
#include <vector>


class PrecisionRecallM
{
public:
    PrecisionRecallM(const char *s);

    void addStartTime();
    void addStopTime();
    void add(std::vector< int > &exact, std::vector< int > &estimate);
    
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


