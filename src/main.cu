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
#include "kernel.hpp"

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
void ftest (int argc, char **argv);


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

    // ftest( argc, argv );

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


// $ bin/gdustdtw exp/Gun_Point_error_3 exp/Gun_Point_error_7
void
exp4 (int argc, char **argv)
{
    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();
    TimeSeriesCollection db2( argv[2], 2, -1 ); // distribution is normal
    db2.normalize();

    TimeSeries t = db.sequences.at(10);
    TimeSeries tt = db.sequences.at(1);
    db.sequences.clear();
    db.sequences.push_back(t);
    db.sequences.push_back(tt);

    GDUST gdust( db );
    DUST  dust( db );
    Watch watch;

    double time_naive = 0;
    double time_multi = 0;
    double time_cpu = 0;

    TimeSeries &ts = db2.sequences[0];

    watch.start();
    gdust.match_naive(ts);
    watch.stop();
    time_naive = watch.getInterval();

    watch.start();
    gdust.match(ts);
    watch.stop();
    time_multi = watch.getInterval();

    // watch.start();
    // dust.match(ts);
    // watch.stop();
    // time_cpu = watch.getInterval();

    std::cout << "naive: " << time_naive << std::endl;
    std::cout << "multi: " << time_multi << std::endl;
    std::cout << "cpu  : " << time_cpu   << std::endl;
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


void ftest (int argc, char **argv)
{
    TimeSeriesCollection db( argv[1], 2, -1 ); // distribution is normal
    db.normalize();

    TimeSeries ts = db.sequences.at(0);

    size_t size = sizeof(float) * 6;

    float *results, *results_GPU, *param, *param_GPU;
    param   = (float*)malloc(size);
    results = (float*)malloc(size);
    cudaMalloc((void**)&param_GPU,   size);
    cudaMalloc((void**)&results_GPU, size);

    for (int i=0; i < ts.length() - 1; i+=2) {
        RandomVariable x = ts.at(i);
        RandomVariable y = ts.at(i+1);

        param[0] = (float)x.distribution;
        param[1] = x.observation;
        param[2] = x.stddev;
        param[3] = (float)y.distribution;
        param[4] = y.observation;
        param[5] = y.stddev;

        cudaMemcpy( param_GPU, param, size, cudaMemcpyHostToDevice );

        for (int j=0; j < 6; j++) {
            results[j] = 0.0f;
        }
        cudaMemcpy( results_GPU, results, size, cudaMemcpyHostToDevice );

        g_f123_test<<< 200, 500 >>>(param_GPU, results_GPU);

        cudaMemcpy( results, results_GPU, size, cudaMemcpyDeviceToHost );

        std::cout << "########################################" << std::endl;
        std::cout << "f1: " << results[0] << ", " << results[1] << std::endl;
        std::cout << "f2: " << results[2] << ", " << results[3] << std::endl;
        std::cout << "f3: " << results[4] << ", " << results[5] << std::endl;
    }


    free(results);
    cudaFree(results_GPU);
    free(param);
    cudaFree(param_GPU);
}
