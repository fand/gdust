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

OPT o;

#define LOOKUPTABLES_PATHNAME "lookuptablesMixed"


int main( int argc, char **argv );
void initOpt( int argc, char **argv );
void checkDistance( int argc, char **argv );
void cleanUp();

void exp1( int argc, char **argv );
void exp2( int argc, char **argv );
void exp3( int argc, char **argv );

int main( int argc, char **argv )
{
    initOpt( argc, argv );
    std::cout.precision( 4 );
    std::cerr.precision( 4 );

    std::cout << "samples: " << INTEGRATION_SAMPLES << std::endl;
    
    // checkDistance( argc, argv );

    exp3( argc, argv );
    
    cleanUp();
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
            double gdustdist = gdust.distance( ts1, ts2, -1 );
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


void exp1( int argc, char **argv )
{
    for (int t = 0; t < 10; t++) {
        char filename[50];
        snprintf(filename, 50, "%s_error_%d", argv[1], (t+1));
        std::cout << filename << std::endl;
        
        TimeSeriesCollection db( filename, 2, -1 ); // distribution is normal
        db.normalize();
    
        DUST dust( db );
        GDUST gdust( db );
    
        
        for (int i = 0; i < 1; i++) {
            TimeSeries &ts1 = db.sequences[i];

            double min_gpu = 100000000;
            double min_gpu_i, min_gpu_j;
            double min_cpu = 100000000;
            double min_cpu_i, min_cpu_j;

            // for (int j = 0; j < db.sequences.size(); j++) {
            for (int j = 0; j < 10; j++) {
                if (i==j) {continue;}
                TimeSeries &ts2 = db.sequences[j];

                double gdustdist = gdust.distance( ts1, ts2, -1 );
                double dustdist = dust.distance( ts1, ts2, -1 );

                if (gdustdist < min_gpu) {
                    min_gpu = gdustdist;
                    min_gpu_j = j;
                }
                if (dustdist < min_cpu) {
                    min_cpu = dustdist;
                    min_cpu_j = j;
                }
            }

            std::cout << "gpu: " << i << " - " << min_gpu_j << std::endl;
            std::cout << "cpu: " << i << " - " << min_cpu_j << std::endl;
        }
        
    }

}

void exp2( int argc, char **argv )
{

    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();
    
    DUST dust( db );
    GDUST gdust( db );
    Watch watch;
    
    double time_gpu = 0;
    double time_cpu = 0;
            
    for (int i = 0; i < 10; i++) {
        TimeSeries &ts1 = db.sequences[rand() % (int)(100)];
        TimeSeries &ts2 = db.sequences[rand() % (int)(100)];

        watch.start();
        double gdustdist = gdust.distance( ts1, ts2, -1 );
        watch.stop();
        time_gpu += watch.getInterval();
                
        watch.start();
        double dustdist = dust.distance( ts1, ts2, -1 );
        watch.stop();
        time_cpu += watch.getInterval();
        // std::cout << "time_cpu: " << time_cpu << std::endl;
    }
        
    std::cout << "gpu: " << time_gpu / 10 << std::endl;
    std::cout << "cpu: " << time_cpu / 10 << std::endl;

}

void exp3( int argc, char **argv )
{

    for (int t = 50; t <= 500; t += 50) {
        char filename[50];
        snprintf(filename, 50, "%s_%d", argv[1], t);
        std::cout << filename << std::endl;
        
        TimeSeriesCollection db( filename, 2, -1 ); // distribution is normal
        db.normalize();
    
        DUST dust( db );
        GDUST gdust( db );
        Watch watch;
    
        double time_gpu = 0;
        double time_cpu = 0;
        double res_gpu = 0;
        double res_cpu = 0;
            
        for (int i = 0; i < 10; i++) {
            TimeSeries &ts1 = db.sequences[rand() % (int)(100)];
            TimeSeries &ts2 = db.sequences[rand() % (int)(100)];

            watch.start();
            double gdustdist = gdust.distance( ts1, ts2, -1 );
            watch.stop();
            time_gpu += watch.getInterval();
            res_gpu += gdustdist;
                
            watch.start();
            double dustdist = dust.distance( ts1, ts2, -1 );
            watch.stop();
            time_cpu += watch.getInterval();
            res_cpu += dustdist;
        }
        
        std::cout << "gdust: " << res_gpu / 10 << std::endl;
        std::cout << "cdust: " << res_cpu / 10 << std::endl;
        std::cout << "time_gpu: " << time_gpu / 10 << std::endl;
        std::cout << "time_cpu: " << time_cpu / 10 << std::endl;
        std::cout << std::endl;
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


