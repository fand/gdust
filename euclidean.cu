#include "euclidean.hpp"
#include "common.hpp"

#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <math.h>


Euclidean::Euclidean( TimeSeriesCollection &collection, bool exact )
{
    this->collection = collection;
    this->exact = exact;
    largestDistanceId = -1;
}


thrust::host_vector< int > Euclidean::rangeQuery( TimeSeries &ts, float threshold )
{
    thrust::host_vector< int > ids;
    
    for( unsigned int i = 0; i < collection.sequences.size(); i++ )
    {
        if( distance( ts, collection.sequences[i] ) <= threshold )
        {
            ids.push_back( i + 1 );
        }
    }

    return ids;
}


float Euclidean::distance( TimeSeries &ts1, TimeSeries &ts2, int n )
{
    if( ts1.length() != ts2.length() )
    {
        FATAL( "ts1 length=" + TO_STR( ts1.length() ) + " != ts2 length=" + TO_STR( ts2.length() ) );
    }

    float dist = 0;

    if( n == -1 )
    {
        n = ts1.length();
    }

    
    if(exact)
    {
        for( int i = 0; i < n; i++ )
        {
            dist += pow( ts1.at(i).groundtruth - ts2.at(i).groundtruth, 2 );
        }
    }
    else
    {
        for( int i = 0; i < n; i++ )
        {
            dist += pow( ts1.at(i).observation -ts2.at(i).observation, 2 );
        }
    }

    dist = sqrt( dist );
    return dist;
}


float Euclidean::getHeuristicThreshold( float abovePercentual )
{
    float histogram[ 1000 ];

    for( int i = 0; i < 1000; i++ )
    {
        histogram[i] = 0;
    }
    
    float min = -1, max = -1;

    for( unsigned int i = 0; i < collection.sequences.size(); i++)
    {
        for( unsigned int j = i+1; j < collection.sequences.size(); j++)
        {
            TimeSeries &t1 = collection.sequences[i];
            TimeSeries &t2 = collection.sequences[j];
            float d = distance(t1, t2);

            if( min == -1 )
            {
                min = d;
                max = d;
            }
            else
            {
                min = MIN( min, d );
                max = MAX( max, d );
            }

        }
    }

    assert( min <= max && min >= 0 );

    float nmax = max - min;

    float alldists = pow( collection.sequences.size(), 2.0 ) / 2.0 - collection.sequences.size();


    for( unsigned int i = 0; i < collection.sequences.size(); i++)
    {
        for( unsigned int j = i+1; j < collection.sequences.size(); j++)
        {
            TimeSeries &t1 = collection.sequences[i];
            TimeSeries &t2 = collection.sequences[j];
            float d = distance( t1, t2 );

            d -= min;
            histogram[ (int)( ceil( d / nmax * (1000 - 1) ) ) ]++;

        }
    }

    unsigned long counter = 0;
    int bucket_id = 0;
    for( bucket_id = 0; bucket_id < 1000; bucket_id++ )
    {
        counter += histogram[ bucket_id ];
        if( counter > alldists*abovePercentual )
        {
            break;
        }
    }

    assert( bucket_id < 1000 );

    float dd = min + ( ( float )bucket_id ) / ( 1000.0 - 1.0 ) * nmax;

    counter = 0;

    for( unsigned int i = 0; i < collection.sequences.size(); i++ )
    {
        for( unsigned int j = i+1; j < collection.sequences.size(); j++ )
        {
            TimeSeries &t1 = collection.sequences[i];
            TimeSeries &t2 = collection.sequences[j];

            if( distance( t1, t2 ) <= dd )
            {
                counter++;
            }
        }
    }

    float coverage = ( ( float )counter ) / alldists;

    assert( coverage - 0.1 < abovePercentual && coverage + 0.1 > abovePercentual );

    return dd;
}


