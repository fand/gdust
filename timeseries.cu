#include "timeseries.hpp"
#include "randomvariable.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <math.h>


TimeSeries::TimeSeries()
{
    this->id = -1;
}


TimeSeries::TimeSeries( const char *path, int distribution )
{
    this->readFile( path, distribution );
}


TimeSeries::TimeSeries( std::string &s, int distribution )
{
    this->readString( s, distribution );
}


void TimeSeries::readString( std::string &s, int distribution )
{
    // clear sequence data before reading string
//    this->sequence.clear();
    this->sequence.empty();
    std::replace( s.begin(), s.end(), ':', ' ' );

    std::stringstream lineStream(s);

    float stddev, observation, groundtruth;

    while( !( lineStream.eof() ) )
    {
        lineStream >> groundtruth;
        lineStream >> observation;
        lineStream >> stddev;
        
        this->sequence.push_back( RandomVariable( distribution, groundtruth, observation, stddev ) );
    }

//    this->sequence_GPU = this->sequence;
}


void TimeSeries::readFile( const char *path, int distribution )
{
    std::ifstream fin( path );
    std::string line;

    if( !getline( fin, line ) )
    {
        std::cout << "Error: cannot open file(" << path << ")" << std::endl;
    }

    this->readString( line, distribution );

    std::cout << "TimeSeries " << path << " : length " << this->sequence.size() << std::endl;
    fin.close();
}


void TimeSeries::normalize()
{
    float average = 0;
    float sampleVariance = 0;

    for( unsigned int i = 0; i < this->sequence.size(); i++ )
    {
        average += this->sequence[i];
    }
    average /= ( float )this->sequence.size();

    for( unsigned int i = 0; i < sequence.size(); i++ )
    {
        sampleVariance += pow( sequence[i] + average, 2 );
    }
    sampleVariance = sqrt( sampleVariance / ( float )this->sequence.size() );

    for( unsigned int i = 0; i< sequence.size(); i++)
    {
        sequence[i] = ( sequence[i] - average ) / sampleVariance;
    }

//    this->sequence_GPU = this->sequence;
}


RandomVariable TimeSeries::at( int index )
{
    return this->sequence[ index ];
}

/*
RandomVariable * TimeSeries::at_GPU( int index )
{
    return &( this->sequence_GPU[ index ] );
}
*/


unsigned int TimeSeries::length()
{
    return this->sequence.size();
}


int TimeSeries::getId()
{
    return this->id;
}


void TimeSeries::setId( int newId )
{
    this->id = newId;
}
