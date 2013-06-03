#include <limits>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
extern char *optarg;
extern int optind, optopt, opterr;

#include "main.hpp"
#include "dataset.hpp"
#include "common.hpp"
#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "euclidean.hpp"
#include "precisionrecallM.hpp"

#include "watch.hpp"

#include "dust.hpp"
#include "gdust.hpp"

#include <cutil.h>

OPT o;

#define LOOKUPTABLES_PATHNAME "lookuptablesMixed"


int main( int argc, char **argv );
void initOpt( int argc, char **argv );
void checkDtw( int argc, char **argv );
void checkDistance( int argc, char **argv );
void checkDistribution( int argc, char **argv );
void checkStddev( int argc, char **argv );
void cleanUp();
void generateFDustTables( int argc, char **argv );




int main( int argc, char **argv )
{
    initOpt( argc, argv );
    std::cout.precision( 4 );
    std::cerr.precision( 4 );
    
//    checkDtw( argc, argv );
    checkDistance( argc, argv );
//    checkDistribution( argc, argv );
//    checkStddev( argc, argv );
    
    cleanUp();
}


void checkDtw( int argc, char **argv )
{

    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    
    std::cout << "input file: " << argv[1] << std::endl;
    
//    db.printSeqs();
    
    
    // MUST DO THIS FIRST!!! (for X-ray data)
    db.normalize();
    
    
    DUST dust( db );
    GDUST gdust( db );
    Euclidean eucl( db );
    
    Watch watch;
    
    std::cout << "eval_loop " << db.sequences.size() << " start" << std::endl;
    
    
    for ( int i = 0; i < db.sequences.size(); i++ )
    {
        TimeSeries &ts1 = db.sequences[i];
        for ( int j = i + 1; j < db.sequences.size(); j++ )
        {
            TimeSeries &ts2 = db.sequences[j];
            
            
            watch.start();
            double gdustdtw = gdust.dtw( ts1, ts2 );
            watch.stop();
            std::cout << i << "-" << j << " gdustdtw :" << gdustdtw 
                      << " time: " << watch.getInterval() << std::endl;
            
            // watch.start();
            // double dustdtw = dust.dtw( ts1, ts2 );
            // watch.stop();
            // std::cout << i << "-" << j << " dustdtw :" << dustdtw 
            //           << " time: " << watch.getInterval() << std::endl;
            
            // watch.start();
            // double eucldtw = eucl.dtw( ts1, ts2 );
            // watch.stop();
            // std::cout << i << "-" << j << " eucldtw :" << eucldtw
            //           << " time: " << watch.getInterval() << std::endl;
        }
    }
}


void checkDistance( int argc, char **argv )
{

    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    
    std::cout << "input file: " << argv[1] << std::endl;
    
//    db.printSeqs();
    
    
    // MUST DO THIS FIRST!!! (for X-ray data)
    db.normalize();
    
    
    DUST dust( db );
    GDUST gdust( db );
    Euclidean eucl( db );
    
    Watch watch;
    
    std::cout << "eval_loop " << db.sequences.size() << " start" << std::endl;    
    
    for ( int i = 0; i < db.sequences.size(); i++ )
    {
        TimeSeries &ts1 = db.sequences[i];
        for ( int j = i + 1; j < db.sequences.size(); j++ )
        {
            TimeSeries &ts2 = db.sequences[j];

            std::cout << "ts1 : " << ts1.length() << std::endl;
            std::cout << "ts1 : " << ts2.length() << std::endl;
            
            watch.start();
            double gdustdist = gdust.distance( ts1, ts2 );
            watch.stop();
            std::cout << i << "-" << j << " gdustdist :" << gdustdist
                      << " time: " << watch.getInterval() << std::endl;
            
            watch.start();
            double dustdist = dust.distance( ts1, ts2 );
            watch.stop();
            std::cout << i << "-" << j << " dustdist :" << dustdist
                      << " time: " << watch.getInterval() << std::endl;
            
            // watch.start();
            // double eucldist = eucl.distance( ts1, ts2 );
            // watch.stop();
            // std::cout << i << "-" << j << " eucldist :" << eucldist
            //           << " time: " << watch.getInterval() << std::endl;
        }
    }
}


void checkDistribution( int argc, char **argv )
{
    TimeSeriesCollection db;
    DUST dust(db);
    GDUST gdust(db);

    for( double distance = 0.0; distance <= 3.0; distance += 0.2 )
    {
        // uniform
        // RandomVariable x = RandomVariable( RANDVAR_UNIFORM, 0, 0, 0.2 );
        // RandomVariable y = RandomVariable( RANDVAR_UNIFORM, distance, distance, 0.8 );
        // double d = dust.dust( x, y );
        // double gd = gdust.dust( x, y );
        // std::cerr << "dust:" << d << " gdust:" << gd << std::endl;

            
        // normal
        RandomVariable x = RandomVariable( RANDVAR_NORMAL, 0, 0, 0.2 );
        RandomVariable y = RandomVariable( RANDVAR_NORMAL, distance, distance, 0.8 );
        double d = dust.dust( x, y );
        double gd = gdust.dust( x, y );
//            std::cerr << "distribution:" << RANDVAR_NORMAL
//                      << " stddevs:" << 0.2 << " "<< 0.8 << std::endl;
        std::cerr << "dust:" << d << " gdust:" << gd << std::endl;
    }
}


void checkStddev( int argc, char **argv )
{
    TimeSeriesCollection db;
    DUST dust(db);
    GDUST gdust(db);
    
    for( double stddev = 0.0; stddev <= 3.0; stddev += 0.2 )
    {
        // uniform
        // RandomVariable x = RandomVariable( RANDVAR_UNIFORM, 0, 0, 0.2 );
        // RandomVariable y = RandomVariable( RANDVAR_UNIFORM, distance, distance, 0.8 );
        // double d = dust.dust( x, y );
        // double gd = gdust.dust( x, y );
        // std::cerr << "dust:" << d << " gdust:" << gd << std::endl;
        
        
        // normal
        RandomVariable x = RandomVariable( RANDVAR_NORMAL, 0, 0, 0.2 );
        RandomVariable y = RandomVariable( RANDVAR_NORMAL, 20, 20, stddev );
        double d = dust.dust( x, y );
        double gd = gdust.dust( x, y );
//            std::cerr << "distribution:" << RANDVAR_NORMAL
//                      << " stddevs:" << 0.2 << " "<< 0.8 << std::endl;
        std::cerr << "dust:" << d << " gdust:" << gd << std::endl;
    }
}



void cleanUp()
{
    SAFE_FREE( o.rfileCollection );
    SAFE_FREE( o.rfileQuery );
    SAFE_FREE( o.rfileQuery2 );
    SAFE_FREE( o.wfile );
}


void  initOpt( int argc, char **argv )
{
    int c;

    o.synthetic = false;
    o.syntheticLength = -1;
    o.rfileCollection = NULL;
    o.rfileQuery = NULL;
    o.rfileQuery2 = NULL;
    o.wfile = NULL;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "C:Q:q:w:" ) ) != EOF )
    {
        switch (c)
        {
        case 'C':
            SAFE_FREE( o.rfileCollection );
            o.rfileCollection = strdup( optarg );
            break;

        case 'q':
            SAFE_FREE( o.rfileQuery );
            o.rfileQuery = strdup( optarg );
            break;
            
        case 'Q':
            SAFE_FREE( o.rfileQuery2 );
            o.rfileQuery2 = strdup( optarg );
            break;
            
        case 'w':
            SAFE_FREE( o.wfile );
            o.wfile = strdup( optarg );
            break;
            
        default:
            FATAL( "option '" + TO_STR( optopt ) + "' invalid" );
        }
    }
}


void generateFDustTables( int argc, char **argv ){
    TimeSeriesCollection db;
    GDUST dust( db );
    dust.buildFDustTables(o.wfile);
}

