#include "gdust.hpp"
#include "integrator.hpp"
#include "common.hpp"

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>

#define VERYSMALL 1E-20


inline float clean_probability( float p )
{
    if(p <= 0) p = 0;
    return p;
}

void GDUST::init()
{
    lookupTablesAvailable = false;
}


float GDUST::phi( RandomVariable &x, RandomVariable &y )
{
    // params to define functions
    float params[6] = {
        (float)x.distribution,
        x.observation,
        x.stddev,
        (float)y.distribution,
        y.observation,
        y.stddev
    };

    float res = this->integ->phi(params);
//    std::cout << "phi: " << res << std::endl;
    return res;
}



GDUST::~GDUST()
{

}


GDUST::GDUST( TimeSeriesCollection &collection, const char *lookUpTablesPath )
{
    this->collection = collection;    
    this->init();

    if( lookUpTablesPath )
    {
        readLookUpTables( lookUpTablesPath );
    }

    this->integ = new Integrator();
}


std::vector< int > GDUST::rangeQuery( TimeSeries ts, float threshold )
{
    std::vector< int > answer;

    for( unsigned int i = 0; i < this->collection.sequences.size(); i++ )
    {
        std::cout << "rangeQuery loop " << i << ", length " << ts.length() << std::endl;

        float dist = 0;
        for( unsigned int j = 0; j < ts.length(); j++ )
        {
            RandomVariable a = ts.at(j);
            RandomVariable b = this->collection.sequences[i].at(j);
//            float tmp = dust( a, b );
//            dist += pow( tmp, 2 );
            dist += pow( dust( a, b ), 2 );            
            if( dist > threshold * threshold )
            {
                break;
            }
        }

        if( sqrt(dist) <= threshold )
        {
            answer.push_back( i+1 );
        }
    }

    return answer;
}


float GDUST::distance( TimeSeries &ts1, TimeSeries &ts2, int n )
{
    float dist = 0;
    
    if( n == -1 ) n = ts1.length();
    if( n > ts2.length() ) n = ts2.length();
    
    for( int i = 0; i < n; i++ )
    {
        RandomVariable a = ts1.at(i);
        RandomVariable b = ts2.at(i);

        dist += pow( dust( a, b ), 2 );
    }

    dist = sqrt( dist );
    return dist;
}


double GDUST::dtw( TimeSeries &ts1, TimeSeries &ts2 )
{
    int len1 = ts1.length();
    int len2 = ts2.length();

    double *table_d = (double*)malloc( sizeof(double) * len1 * len2 );
    double *table_g = (double*)malloc( sizeof(double) * len1 * len2 );

    calcCost( ts1, ts2, table_d, table_g, len1, len2 );
    calcGamma( table_d, table_g, len1, len2 );
    double dist = calcSum(table_d, table_g, len1, len2 );

//    dist = sqrt( dist );

    free( table_d );
    free( table_g );
    return dist;
}

void GDUST::calcCost( TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2 )
{
    for ( int i = 0; i < len1; i++ )
    {
        RandomVariable r1 = ts1.at(i);

        for ( int j = 0; j < len2; j++ )
        {
            RandomVariable r2 = ts2.at(j);
//            table_d[ i*len2 + j ] = pow( dust( r1, r2 ), 2 );
            table_d[ i*len2 + j ] = dust( r1, r2 );
        }
    }
}

void GDUST::calcGamma( double *table_d, double *table_g, int len1, int len2 )
{
    table_g[0] = table_d[0];
    
    for ( int i = 1; i < len1; i++ )
    {
        table_g[ i*len2 ] = table_d[ i*len2 ] + table_g[ (i-1) * len2 ];
    }

    for ( int i = 1; i < len2; i++ )
    {
        table_g[i] = table_d[i] + table_g[i-1];
    }
    
    for ( int i = 1; i< len1; i++ )
    {
        for( int j = 1; j < len2; j++ )
        {
            table_g[ i*len2 + j ] =
                table_d[ i*len2 + j ] +
                std::min(
                    std::min(
                        table_g[ (i-1) * len2 + j ],
                        table_g[ i*len2 + j-1 ]
                    ),
                    table_g[ (i-1)*len2 + j-1 ]
                );
        }
    }
}

double GDUST::calcSum( double *table_d, double *table_g, int len1, int len2 )
{
    double sum = 0.0;
    
    int i = len1 - 1;
    int j = len2 - 1;
    
    while( i > 0 || j > 0 ){
        
        sum += table_g[ i*len2 + j ];
            
        if( i == 0 ){
            j--;
        }else if( j == 0 ){
            i--;
        }else{
            double m = std::min(
                std::min(
                    table_g[ (i-1)*len2 + j ],
                    table_g[ i*len2 + j-1 ] ),
                table_g[ (i-1)*len2 + j-1 ]
            );
            if( m == table_g[ (i-1)*len2 + j ] ){
                i--;
            }else if( m == table_g[ i*len2 + j-1 ] ){
                j--;
            }else{
                i--; j--;
            }
        }
    }

    return sum;
}




float GDUST::dust( RandomVariable &x, RandomVariable &y )
{
    float distance = -log10( phi(x,y) );
    if (distance < 0) {
        distance = 0;
    }
    distance = sqrt( distance );

//     assert( distance != 0 );
    return distance;
}


void GDUST::buildFDustTables( const char *path )
{
    std::ofstream fout( path );

    if( ! fout )
    {
        FATAL( "error opening outputfile." );
    }

    fout.precision( 4 );

    for( int distribution = 1; distribution <= 3; distribution++ )
    {
        for( float stddev0 = STDDEV_BEGIN; stddev0 < STDDEV_END+STDDEV_STEP/2.0; stddev0+=STDDEV_STEP)
        {
            for( float stddev1 = stddev0; stddev1 < STDDEV_END+STDDEV_STEP/2.0; stddev1+=STDDEV_STEP)
            {
                for( float distance = DISTANCE_BEGIN; distance < DISTANCE_END+DISTANCE_STEP/2.0; distance+= DISTANCE_STEP )
                {

                    RandomVariable x = RandomVariable( distribution, (float)0, (float)0, stddev0 );
                    RandomVariable y = RandomVariable( distribution, distance, distance, stddev1 );

                    float d;

                    d = dust(x, y);

                    std::cerr << "distribution:" << distribution << " stddev0:" << stddev0 << " stddev1:" << stddev1 << " distance:" << distance << " dustdistance:" << d << std::endl;

                    fout << distribution << " " << stddev0 << " " << stddev1 <<  " " << distance << " " << d << "\n";
                    fout << distribution << " " << stddev1 << " " << stddev0 <<  " " << distance << " " << d << "\n";
                }
            }
        }
    }
}


void GDUST::readLookUpTables( const char *lookUpTablesPath )
{
    std::ifstream fin( lookUpTablesPath );

    if( fin == NULL )
    {
        FATAL("unable to read lookuptables from file");
    }

    std::string line;
    
    for( int i = 0; i < 3; i++ )
    {
        for( int j0 = 0; j0 < STDDEV_STEPS; j0++ )
        {
            for( int j1 = 0; j1 < STDDEV_STEPS; j1++ )
            {
                for( int k = 0; k < DISTANCE_STEPS; k++ )
                {
                    lookuptables[ i][ j0 ][ j1 ][ k ] = -1;
                }
            }
        }
    }

    while( getline( fin, line ) )
    {
        std::stringstream lineStream( line );

        int distribution;
        float stddev0,stddev1, distance, dustdistance;

        lineStream >> distribution;
        lineStream >> stddev0;
        lineStream >> stddev1;
        lineStream >> distance;
        lineStream >> dustdistance;

        assert( distribution >= 1 && distribution <= 3 );
        assert( stddev0 >= 0 && stddev0 <= STDDEV_END );
        assert( stddev1 >= 0 && stddev1 <= STDDEV_END );
        assert( distance >= 0 && distance <= DISTANCE_END );
        assert( dustdistance >= 0 );

        int stddev0_offset = (int)round( stddev0 / STDDEV_STEP );
        int stddev1_offset = (int)round( stddev1 / STDDEV_STEP );
        int distance_offset = (int)round( distance / DISTANCE_STEP );

        assert(stddev0_offset >= 0 && stddev0_offset < STDDEV_STEPS);
        assert(stddev1_offset >= 0 && stddev1_offset < STDDEV_STEPS);
        assert(distance_offset >= 0 && distance_offset < DISTANCE_STEPS);

        lookuptables[ distribution - 1 ][ stddev0_offset ][ stddev1_offset ][ distance_offset ] = dustdistance;
    }

    fin.close();

    for( int i = 0; i < 3; i++ )
    {
        for(int j0 = 0; j0 < STDDEV_STEPS; j0++)
        {
            for(int j1 = 0; j1 < STDDEV_STEPS; j1++)
            {
                for(int k = 0; k < DISTANCE_STEPS; k++)
                {
                    assert(
                        lookuptables[ i ][ j0 ][ j1 ][ k ] >= 0
                    ); // this ensures that all needed values are present.
                }
            }
        }
    }

    lookupTablesAvailable = true;
}
