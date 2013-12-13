#pragma once

#include "randomvariable.hpp"

#include <string>
#include <vector>


class TimeSeries
{
public:
    TimeSeries();
    TimeSeries(std::vector<RandomVariable> seq);    
    TimeSeries( const char *path, int distribution );
    TimeSeries( std::string &s, int distribution );
    void readFile( const char *path, int distribution );
    void readString( std::string &s, int distribution );
    RandomVariable at( int index );
    
    unsigned int length();
    int getId();
    void setId( int newId );
    void normalize();
    void printSeq();
    
private:
    std::vector< RandomVariable > sequence;
    int id;
};

