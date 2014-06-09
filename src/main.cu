#include "main.hpp"
#include "dataset.hpp"
#include "common.hpp"
#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "precisionrecallM.hpp"
#include "euclidean.hpp"
#include "dust.hpp"
#include "gdust.hpp"
#include "watch.hpp"
#include "config.hpp"

#include <limits>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
extern char *optarg;
extern int optind, optopt, opterr;

OPT o;

#define LOOKUPTABLES_PATHNAME "lookuptablesMixed"



int  main (int argc, char **argv);
void initOpt (int argc, char **argv);
void checkDistance (int argc, char **argv);
void cleanUp();

void exp1 (int argc, char **argv);
void exp2 (int argc, char **argv);
void exp3 (int argc, char **argv);
void exp4 (int argc, char **argv);



int
main (int argc, char **argv)
{
    initOpt( argc, argv );
    std::cout.precision( 4 );
    std::cerr.precision( 4 );

    // checkDistance( argc, argv );

    // exp1( argc, argv );
    // exp2( argc, argv );
    // exp3( argc, argv );
    exp4( argc, argv );
    
    cleanUp();
}


void
checkDistance (int argc, char **argv)
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


void
exp1 (int argc, char **argv)
{
    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();
    
    DUST dust( db );
    GDUST gdust( db );
    Watch watch;
    
    double time_gpu = 0;
    double time_cpu = 0;
            
    for (int i = 0; i < 9; i++) {
        for (int j = i+1; j < 10; j++) {
            TimeSeries &ts1 = db.sequences[i];
            TimeSeries &ts2 = db.sequences[j];

            watch.start();
            double gdustdist = gdust.distance( ts1, ts2, -1 );
            watch.stop();
            time_gpu += watch.getInterval();
                
            watch.start();
            double dustdist = dust.distance( ts1, ts2, -1 );
            watch.stop();
            time_cpu += watch.getInterval();
        }
    }
        
    std::cout << "gpu: " << time_gpu / 45.0 << std::endl;
    std::cout << "cpu: " << time_cpu / 45.0 << std::endl;
}


void
exp2 (int argc, char **argv)
{
    for (int t = 50; t <= 500; t += 50) {
        char filename[50];
        snprintf(filename, 50, "%s/exp2/Gun_Point_error_3_trunk_%d", argv[1], t);
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

       for (int i = 0; i < 9; i++) {
           for (int j = i; j < 10; j++) {
                // TimeSeries &ts1 = db.sequences[rand() % (int)(100)];
                // TimeSeries &ts2 = db.sequences[rand() % (int)(100)];
                TimeSeries &ts1 = db.sequences[i];
                TimeSeries &ts2 = db.sequences[j];

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
        }
        
        std::cout << "gdust: " << res_gpu / 45 << std::endl;
        std::cout << "cdust: " << res_cpu / 45 << std::endl;
        std::cout << "time_gpu: " << time_gpu / 45 << std::endl;
        std::cout << "time_cpu: " << time_cpu / 45 << std::endl;
        std::cout << std::endl;
    }
}

void
exp3 (int argc, char **argv)
{
    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();
    
    DUST dust( db );
    GDUST gdust( db );
    Watch watch;
    
    double time_gpu = 0;
    double time_cpu = 0;
            
    for (int i = 0; i < 9; i++) {
        for (int j = i+1; j < 10; j++) {
            TimeSeries &ts1 = db.sequences[i];
            TimeSeries &ts2 = db.sequences[j];

            watch.start();
            double gdustdist = gdust.distance( ts1, ts2, -1 );
            watch.stop();
            time_gpu += watch.getInterval();
                
            watch.start();
            double dustdist = dust.distance( ts1, ts2, -1 );
            watch.stop();
            time_cpu += watch.getInterval();
        }
    }
        
    std::cout << "gpu: " << time_gpu / 45.0 << std::endl;
    std::cout << "cpu: " << time_cpu / 45.0 << std::endl;
}


void
exp4 (int argc, char **argv)
{
    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();
    
    GDUST gdust( db );
    Watch watch;
    
    double time_gpu = 0;

    TimeSeries &ts = db.sequences[0];
    
    watch.start();    
    gdust.match_naive(ts);
    watch.stop();
    time_gpu = watch.getInterval();
                        
    std::cout << "gpu: " << time_gpu << std::endl;
}


void
cleanUp()
{
    SAFE_FREE( o.rfileCollection );
    SAFE_FREE( o.rfileQuery );
    SAFE_FREE( o.rfileQuery2 );
    SAFE_FREE( o.wfile );
}


void
initOpt (int argc, char **argv)
{
    int c;

    o.synthetic = false;
    o.syntheticLength = -1;
    o.rfileCollection = NULL;
    o.rfileQuery = NULL;
    o.rfileQuery2 = NULL;
    o.wfile = NULL;

    opterr = 0;

    while ((c = getopt( argc, argv, "C:Q:q:w:" )) != EOF) {
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


