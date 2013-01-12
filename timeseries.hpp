#pragma once

#include "randomvariable.hpp"

#include <string>
#include <thrust/host_vector.h>
#include <vector>

class TimeSeries
{
public:
    TimeSeries();
    TimeSeries( const char *path, int distribution );
    TimeSeries( std::string &s, int distribution );
    void readFile( const char *path, int distribution );
    void readString( std::string &s, int distribution );
    void normalize();
    RandomVariable at( int index );
//    RandomVariable * at_GPU( int index );
    unsigned int length();
    int getId();
    void setId( int newId );
    
private:
//    thrust::host_vector< RandomVariable > sequence;
//    thrust::device_vector< RandomVariable > sequence_GPU;
    std::vector< RandomVariable > sequence;
    int id;
};

