#include "dataset.hpp"
#include "common.hpp"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

DataSet::DataSet()
{
}

void DataSet::randomWalks( long N, long length )
{
    boost::mt19937 rng;
    boost::uniform_real<> ur( 0.0, 1.0 );
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > drawSample( rng, ur );

    this->N = N;

    for( int i = 0; i < N; i++ )
    {
        this->originalTimeSeries[i].resize( length );
        for( int j = 0; j < length; j++ )
        {
            this->originalTimeSeries[i][j] = 100.0;
            for( int k = 0; k < j; k++)
            {
                this->originalTimeSeries[i][j] += drawSample() - 0.5;
            }
        }
    }
}

void DataSet::readFile( const char *src )
{
    std::ifstream fin( src );

    std::cerr << "Reading real timeseries collection from file : " << src << std::endl;
    
    std::string line;
    int i = 0;

    while( std::getline( fin, line ) )
    {
        if( i >= TIMESERIES_MAX )
        {
            FATAL( "too many timeseries in this collection" );
        }
        
        std::stringstream lineStream( line );
        float v;
        
        while( ! lineStream.eof() )
        {
            lineStream >> v;
            if( ! lineStream.eof())
            {
                this->originalTimeSeries[i].push_back( v );
            }
        }

        normalize( originalTimeSeries[i] );

        std::cerr << "TimeSeries " << i+1 << " has length " << originalTimeSeries[i].size() << std::endl;
        i++;
    }

    this->N = i;

    std::cerr << "Done." << std::endl;

    fin.close();
}

void DataSet::normalize( thrust::host_vector< float > &timeSeries )
{
    float average = 0;
    float maxdev = 0;

    for( unsigned int i = 0; i < timeSeries.size(); i++ )
    {
        average += timeSeries[i];
    }
    average /= ( float )timeSeries.size();

    for( unsigned int i = 0; i < timeSeries.size(); i++ )
    {
        maxdev = MAX( maxdev, ABS( average - timeSeries[i] ) );
    }

    for( unsigned int i = 0; i < timeSeries.size(); i++ )
    {
        timeSeries[i] = ( timeSeries[i] - average ) / maxdev;
    }
}

void DataSet::writeMultiSamplesDir( const char *dst )
{
    std::cerr << "Writing all timeseries to separate files in dir : " << dst << std::endl;
    
    for( int i = 0; i < this->N; i++ )
    {
        for( int j = 0; j < SAMPLES_MAX; j++ )
        {
            normalize( perturbatedTimeSeries[i][j] );
        }
    }

    if( mkdir( dst, S_IRUSR | S_IWUSR | S_IXUSR ) )
    {
        FATAL( "mkdir: " + std::string( strerror( errno ) ) );
    }

    char name[ 512 ];

    for( int i = 0; i < this->N; i++ )
    {
        sprintf( name, "%s/timeseries%d", dst, i+1 );


        std::ofstream fout( name );
        
        if( ! fout )
        {
            FATAL( "error opening output file : " + std::string( name ) + std::string( strerror( errno ) ) );
        }
        
        fout.precision( 4 );
        
        for( unsigned int k = 0; k < originalTimeSeries[ i ].size(); k++ )
        {
            for( unsigned int j = 0; j < SAMPLES_MAX; j++ )
            {
                fout << perturbatedTimeSeries[i][j][k] << ( j == ( SAMPLES_MAX - 1 ) ? "" : " " );
            }
            
            fout << std::endl;
        }
        
        fout.close();
    }

    std::cerr << "Done.\n";
}


void DataSet::perturbateNormal( float mu, float sigma )
{
    boost::mt19937 rng( static_cast< unsigned >( time(0) ) );
    boost::normal_distribution<> nd( mu, sigma );

//    boost::variate_generator< boost::mt19937&, boost::normal_distribution<> > drawSample( rng, nd );
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> > drawSample( rng, nd );    

    std::cerr << "adding perturbation (normal) ..." << std::endl;

    for( int i = 0; i < this->N; i++ )
    {
        std::cerr << "processing TimeSeries " << i + 1 << " ...\n";

        for( int j = 0; j < SAMPLES_MAX; j++ )
        {
            perturbatedTimeSeries[i][j].resize( originalTimeSeries[i].size() );
        }

        for( int k = 0; k < this->N; k++ )
        {
            for( int l = 0; l < SAMPLES_MAX; l++ )
            {
                for( unsigned int m = 0; m < perturbatedTimeSeries[k][l].size(); m++ )
                {
                    perturbatedTimeSeries[k][l][m] = originalTimeSeries[k][l] + drawSample();
                }
            }
        }
    }
}
