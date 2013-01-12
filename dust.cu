#include "dust.hpp"
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

void DUST::init()
{
    lookupTablesAvailable = false;
}


float DUST::phi( RandomVariable &x, RandomVariable &y )
{
    RandomVariable *pair[2];
    pair[0] = &x;
    pair[1] = &y;

    // params to define functions
    float params[8] = {
        0.0f,
        0.0f,
        (float)x.distribution,
        x.observation,
        x.stddev,
        (float)y.distribution,
        y.observation,
        y.stddev
    };



    // p(x|r(x)=v)p(r(x)=v) and p(y|r(y)=v)p(r(y)=v)
    float int1 = clean_probability( (this->integ)->integrate( 1, params ) );
    params[0] = int1;

    float int2 = clean_probability( (this->integ)->integrate( 2, params ) );
    params[1] = int2;
    
    if( int1 > 1.0 )
    {
        std::cout << "int1 error!" << int1 << std::endl;
        exit(1);
    }

    if(int1 == 0) int1 = VERYSMALL;
    if(int2 == 0) int2 = VERYSMALL;    

    // p(r(x)=r(y)|x,y)
    // pdf for "real x == real y"
    float int3 = (this->integ)->integrate( 3, params );
    int3 = clean_probability( int3 );


    if( false ) {
        std::cerr << "xdistrib:" << x.distribution << " ydistrib:" << y.distribution << std::endl;
        std::cerr << "int1:" << int1 << " int2:" << int2 << " int3:" << int3 << std::endl;
    }
//    assert( NOTNANINF(int3) );

    return int3;
}



DUST::~DUST()
{

}


DUST::DUST( TimeSeriesCollection &collection, const char *lookUpTablesPath )
{
    this->collection = collection;    
    this->init();

    if( lookUpTablesPath )
    {
        readLookUpTables( lookUpTablesPath );
    }

    this->integ = new Integrator();
}


thrust::host_vector< int > DUST::rangeQuery( TimeSeries ts, float threshold )
{
    thrust::host_vector< int > answer;

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


float DUST::distance( TimeSeries &ts1, TimeSeries &ts2, int n )
{
    float dist = 0;
    
    if( n == -1 ) n = ts1.length();
    
    for( int i = 0; i < n; i++ )
    {
        RandomVariable a = ts1.at(i);
        RandomVariable b = ts2.at(i);

        dist += pow( dust( a, b ), 2 );
    }

    dist = sqrt( dist );
    return dist;
}


float DUST::dust( RandomVariable &x, RandomVariable &y )
{
    float distance = -log10( phi(x,y) );
    if ( distance < 0 ) distance = 0; 
    return sqrt(distance);
}


void DUST::buildFDustTables( const char *path )
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


void DUST::readLookUpTables( const char *lookUpTablesPath )
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
