#include "timeseriescollection.hpp"
#include "euclidean.hpp"
#include "common.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <algorithm>


TimeSeriesCollection::TimeSeriesCollection()
{
}


TimeSeriesCollection::TimeSeriesCollection( const char *path, int distribution, int limitN )
{
    this->readFile( path, distribution, limitN );
}


void TimeSeriesCollection::readFile( const char *path, int distribution, int limitN )
{
    std::ifstream fin(path);

    if(fin == NULL)
    {
        FATAL( "unable to read timeseriescollection from file" );
    }

    std::string line;
    while( getline( fin, line ) )
    {
        sequences.push_back( TimeSeries( line, distribution ) );
        sequences[ sequences.size() - 1 ].setId( sequences.size() );

        if( limitN != -1 && (int)sequences.size() == limitN )
        {
            break;
        }
    }

    std::cerr << "Read " << sequences.size()
              << " timeseries from " << path
              << ", sequenceslength =  " << sequences.size() << std::endl;
//              << ", length=" << sequences[ 0 ].length() << std::endl;

    fin.close();
}


struct mycompare
{
    bool operator()( PairIdDistance const &a, PairIdDistance const &b )
    {
        return a.second < b.second;
    }
};


void TimeSeriesCollection::computeTopKthresholds(
    unsigned int K,
    thrust::host_vector< PairIdDistance > &topIdDists
)
{
    assert( sequences.size() >= K );
    topIdDists.resize( sequences.size() );

    std::cout << "resized" << std::endl;
    
    Euclidean trueEuclidean( *this, true );

    for( unsigned int i = 0; i < sequences.size(); i++)
    {
        TimeSeries &q = sequences[i];
        thrust::host_vector< PairIdDistance > distances;

        for( unsigned int j = 0; j < sequences.size(); j++)
        {
            TimeSeries &t = sequences[j];
            PairIdDistance d = PairIdDistance( t.getId(), trueEuclidean.distance( q, t ) );
            distances.push_back( d );
        }

        std::sort( distances.begin(), distances.end(), mycompare() );
        topIdDists[ q.getId() - 1 ] = distances[ K ];
    }
}


void TimeSeriesCollection::normalize()
{
    for (int i=0; i<sequences.size(); i++) {
        sequences[i].normalize();
    }
}
void TimeSeriesCollection::printSeqs()
{
    for (int i=0; i<sequences.size(); i++) {
        sequences[i].printSeq();
    }
}
