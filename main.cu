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
#include "dust.hpp"
#include "timeseries.hpp"
#include "timeseriescollection.hpp"
#include "euclidean.hpp"
#include "precisionrecallM.hpp"
#include <cassert>


#include <cutil.h>

OPT o;

#define LOOKUPTABLES_PATHNAME "lookuptablesMixed"

const char *names[] = {
    "50words",
    "Adiac",
    "Beef",
    "CBF",
    "Coffee",
    "ECG200",
    "FaceAll",
    "FaceFour",
    "FISH",
    "Gun_Point",
    "Lighting2",
    "Lighting7",
    "OliveOil",
    "OSULeaf",
    "SwedishLeaf",
    "synthetic_control",
    "Trace",
    NULL
};

bool active[] = {
    false, //    "50words",
    false, //   "Adiac",
    false, //   "Beef",
    false, //   "CBF",
    true, //   "Coffee",
    false, //   "ECG200",
    false, //   "FaceAll",
    false, //   "FaceFour",
    false, //   "FISH",
    false, //   "Gun_Point",
    false, //    "Lighting2",
    false, //    "Lighting7",
    false, //   "OliveOil",
    false, //  "OSULeaf",
    false, //   "SwedishLeaf",
    false,  //   "synthetic_control",
    false //  "Trace", // 17 datasets
};


int main( int argc, char **argv );
void initOpt( int argc, char **argv );
void cleanUp();
void experiment1( TimeSeriesCollection &db, float threshold );
void experiment0( TimeSeriesCollection &db );
void experiment2();
void experiment3( TimeSeriesCollection &db );
void experiment4();
void experiment5();
void experiment6( int argc, char **argv );
void generateFDustTables( int argc, char **argv );




int main( int argc, char **argv )
{
    initOpt( argc, argv );
    std::cout.precision( 4 );
    std::cerr.precision( 4 );

    
    experiment6( argc, argv );

    if(0)
    {
        TimeSeriesCollection db;
        DUST dust( db );

        for( float distance = 0.0; distance <= 10.0; distance += 0.1 )
        {
            RandomVariable x = RandomVariable( RANDVAR_NORMAL, 0, 0, 0.2 );
            RandomVariable y = RandomVariable( RANDVAR_NORMAL, distance, distance, 0.8 );
            float d = dust.dust( x, y );
            std::cerr << "distribution:" << RANDVAR_NORMAL << " stddevs:" << 0.2 << " "<< 0.8  << " distance:" << distance << " dustdistance:" << d << std::endl;
        }
    }

    if(0)
    {
        TimeSeriesCollection db;
        DUST dust( db );

        float distribution = RANDVAR_UNIFORM;

        for( float stddev = 0.2; stddev <= 0.8; stddev += 0.2 )
        {
            for( float distance = 0.0; distance <= 10.0; distance += 0.1 )
            {
                RandomVariable x = RandomVariable( distribution, 0, 0, stddev );
                RandomVariable y = RandomVariable( distribution, distance, distance, stddev);
                float d = dust.dust(x, y);
                std::cerr << "distribution:" << distribution << " stddev:" << stddev << " distance:" << distance << " dustdistance:" << d << std::endl;
            }
        }
    }
    cleanUp();
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


void experiment0( TimeSeriesCollection &db )
{
    Euclidean euclideanTrue( db, true );

    if(0)
    {
        for( unsigned int i = 0; i < db.sequences.size(); i++ )
        {
            for( unsigned int j = i+1; j < db.sequences.size(); j++ )
            {
                TimeSeries &t1 = db.sequences[i];
                TimeSeries &t2 = db.sequences[j];
                std::cout << "euclNormTrue distance(" << t1.getId() << "," << t2.getId() << ")=" << euclideanTrue.distance(t1, t2) << std::endl;
            }
        }
    }

    float desiredRatio = 0.10;
    float threshold = euclideanTrue.getHeuristicThreshold( desiredRatio );
    std::cout << "threshold returning " << desiredRatio << " of the dataset: " << threshold << std::endl;
}


void experiment1( int argc, char **argv, TimeSeriesCollection &db, float threshold )
{
    Euclidean euclideanTrue( db, true );
    Euclidean euclidean( db );
    DUST dust( db, "ltnormal" );

    PrecisionRecallM prEuclidean( "Euclidean" );
    PrecisionRecallM prDust( "DUST" );

    std::cout << "experiment: performing " << db.sequences.size() << " range queries, averaging results on " << 1  << " runs." << std::endl;

    for( unsigned int i = 0; i < db.sequences.size(); i++ )
    {
        TimeSeries &query = db.sequences[i];

        std::cout << "[" << i+1 <<  "/" << db.sequences.size() << "]" << std::endl;

        thrust::host_vector< int > exact = euclideanTrue.rangeQuery( query, threshold );
        thrust::host_vector< int > r;

        float dustThreshold =
            dust.distance( query,
                           db.sequences[ euclideanTrue.largestDistanceId - 1 ] );
        float euclideanThreshold =
            euclidean.distance( query,
                                db.sequences[ euclideanTrue.largestDistanceId - 1 ] );

        r = euclidean.rangeQuery( query, euclideanThreshold );
        prEuclidean.add( exact, r );

        r = dust.rangeQuery( query, dustThreshold );
        prDust.add( exact, r );
    }

    prEuclidean.print();
    prDust.print();
}


void experiment2()
{
    TimeSeriesCollection db( o.rfileCollection, 2 );
    Euclidean euclideanTrue( db, true );
    Euclidean euclidean( db );

    float normalization = db.sequences[0].length();

    for( unsigned int i = 0; i < db.sequences.size(); i++ )
    {
        for( unsigned int j = i+1; j < db.sequences.size(); j++ )
        {
            TimeSeries &t1 = db.sequences[i];
            TimeSeries &t2 = db.sequences[j];
            std::cout << "normalizedDistance(" << t1.getId() << "," << t2.getId() << "): "
                "EuclTrue=" << euclideanTrue.distance( t1, t2 ) / normalization  << " " <<
                "Eucl=" << euclidean.distance( t1, t2 ) / normalization << " " << std::endl;
        }
    }
}

void experiment3( int argc, char **argv, TimeSeriesCollection &db )
{
    DUST dust( db );
    Euclidean euclidean( db );

    for( unsigned int i = 0; i < db.sequences[0].length(); i++ )
    {
        float d1 = euclidean.distance( db.sequences[ 0 ],
                                        db.sequences[ 200 ],
                                        i + 1 );
        float d2=  dust.distance( db.sequences[ 0 ],
                                   db.sequences[ 200 ],
                                   i + 1);
        std::cout << "distance: " << i+1 << " " << d1 << " " << d2 << std::endl;
    }
}

void experiment4_evaluate(
    TimeSeriesCollection &db,
    DUST &dust,
    thrust::host_vector< PairIdDistance > &topIdDists,
    PrecisionRecallM &pr,
    PrecisionRecallM &prEuclidean,
    PrecisionRecallM &prRand
){
    Euclidean euclideanTrue(db, true);
    Euclidean euclidean(db);

    for( unsigned int i = 0; i < db.sequences.size(); i++ )
    {
        TimeSeries &query = db.sequences[i];
        float threshold = topIdDists[ query.getId() - 1 ].second;

        thrust::host_vector< int > exact = euclideanTrue.rangeQuery( query, threshold );
        thrust::host_vector< int > r;

        std::cout << "lets dustdistance!" << std::endl;
        float dustThreshold =
            dust.distance( query,
                           db.sequences[ topIdDists[ query.getId() - 1 ].first - 1 ] );
        std::cout << "lets eucldistance!" << std::endl;
        float euclideanThreshold =
            euclidean.distance( query,
                                db.sequences[ topIdDists[ query.getId() - 1 ].first - 1 ] );

        std::cout << "DUSTdist is " << dustThreshold << ", EUCLdist is " << euclideanThreshold << std::endl;
        
        std::cout << "trimming dust..." << std::endl;
        pr.addStartTime();
        r = dust.rangeQuery( query, dustThreshold );
        pr.addStopTime();
        pr.add( exact, r );
//        assert( ! isnan( pr.getF1() ) );

        std::cout << "trimming EUCL..." << std::endl;        
        prEuclidean.addStartTime();
        r = euclidean.rangeQuery( query, euclideanThreshold );
        prEuclidean.addStopTime();
        prEuclidean.add( exact, r );

        int size = r.size(); // used by prRand

        // now, we put |r| random ids in prRand:
/*/        {
            r.clear();
            mt19937 rng( static_cast< unsigned > (time(0)) );
            uniform_int<> dist( 0, db.sequences.size() - 1 );
            variate_generator< boost::mt19937&, boost::uniform_int<> > drawSample( rng, dist );
            std::cout << "using random ids, ";
            for( int i = 0; i < size; i++ )
            {
                int id = drawSample() + 1;
                r.push_back(id);
            }

            prRand.add( exact, r );
            prRand.addStartTime();
            prRand.addStopTime();
//        }
*/
        assert( ! isnan( prEuclidean.getF1() ) );
        std::cout << "loop " << i << " end" << std::endl;
    }
}

void experiment4( int argc, char **argv )
{
    char inputPathname[ 1024 ];
    char outputPathname[ 1024 ];
    const char *dustlookupTables = LOOKUPTABLES_PATHNAME;

    const char *distribs[] = { "uniform",
                               "normal",
                               "exponential",
                               NULL };

    for(int i = 0; names[i]; i++)
    {
        thrust::host_vector< PairIdDistance > topIdDists;
        bool initializeIdDists = true;

        for( int j = 0; distribs[j]; j++ )
        {
            // XXX just normal is enabled.
            if( j != 1 ) continue;

            sprintf( outputPathname,
                     "results/%s_dust_%s.dat",
                     names[i],
                     distribs[j]);

            std::cout << "output file: " << outputPathname << std::endl;
            
            std::ofstream fout( outputPathname );
            if( ! fout )
            {
                FATAL( "error opening outputfile" );
            }
            fout.precision( 4 );

            for( float stddev = STDDEV_BEGIN; stddev <= STDDEV_END+STDDEV_STEP/2.0; stddev += STDDEV_STEP )
            {

                std::cout << "Processing Dataset:" << names[i] << " distrib:" << distribs[j] << " stddev:" << stddev << std::endl;
                sprintf( inputPathname,
                         "udatasets/%s/%s_dust_%s_%.1f_.dat",
                         names[i],
                         names[i],
                         distribs[j],
                         stddev );

                TimeSeriesCollection db( inputPathname, j + 1 );

                std::cout << "input file: " << inputPathname << std::endl;

                if( initializeIdDists )
                {
                    db.computeTopKthresholds( 10, topIdDists );
                    initializeIdDists = false;
                }

                DUST dust( db, dustlookupTables );
                PrecisionRecallM pr( "DUST" );
                PrecisionRecallM prEuclidean( "Euclidean" );
                PrecisionRecallM prRand( "RANDOM" );
                PrecisionRecallM prMoving( "Moving" );

                experiment4_evaluate( db,
                                      dust,
                                      topIdDists ,
                                      pr,
                                      prEuclidean,
                                      prRand );
                
                fout << j << " " << stddev << " "
                     << pr.getPrecision() <<  " "
                     << pr.getRecall() << " "
                     << pr.getF1() << " "
                     << pr.getTime() << " "
                     << prEuclidean.getPrecision() << " "
                     << prEuclidean.getRecall() << " "
                     << prEuclidean.getF1() << " "
                     << prEuclidean.getTime() << " "
                     << prRand.getPrecision() << " "
                     << prRand.getRecall() << " "
                     << prRand.getF1() << " "
                     << prMoving.getPrecision() << " "
                     << prMoving.getRecall() << " "
                     << prMoving.getF1() << " " << prMoving.getTime() <<  std::endl;

                fout.flush();
            }

            fout.close();
        }
    }
}

void experiment5()
{
    char inputPathname[ 1024 ];

    for( int i = 0; names[i]; i++ )
    {
        sprintf( inputPathname,
                 "udatasets/%s/%s_dust_%s_%.1f_.dat",
                 names[i],
                 names[i],
                 "uniform",
                 0.2);
        TimeSeriesCollection db( inputPathname, 1 );
        Euclidean euclideanTrue( db, true );
        float desiredRatio = 0.1;
        float threshold = euclideanTrue.getHeuristicThreshold( desiredRatio );
        std::cout << "dataset: " << names[i] << " desiredRatio:" << desiredRatio << " => threshold: " << threshold << std::endl;
    }
}

void experiment6( int argc, char **argv )
{
    char inputPathname[ 1024 ];
    char outputPathname[ 1024 ];
    const char *dustlookupTables = LOOKUPTABLES_PATHNAME;

    for(int i = 0; names[i]; i++)
    {
        if( ! active[i] ) continue;

        thrust::host_vector< PairIdDistance > topIdDists;
        bool initializeIdDists = true;

        sprintf( outputPathname,
                 "./results/%s_dust_exponential_mixed.dat",
                 names[i] );

        std::cout << "output file: " << outputPathname << std::endl;
        std::ofstream fout( outputPathname );
        if( ! fout )
        {
            FATAL( "error opening outputfile" );
        }
        fout.precision( 4 );

        std::cout << "processing dataset:" << names[i] << " distrib: .. stddev: mixed" << std::endl;
        sprintf( inputPathname,
                 "udatasets/%s/%s_dust_exponential_mixed_.dat",
                 names[i],
                 names[i] );


        TimeSeriesCollection db( inputPathname, 2, -1 ); // XXX fixing distribution to normal.

        std::cout << "input file: " << inputPathname << std::endl;

        if( initializeIdDists )
        {
            db.computeTopKthresholds( 10, topIdDists );
            initializeIdDists = false;
        }

//        DUST dust( db, dustlookupTables );
        DUST dust( db );
        PrecisionRecallM pr( "DUST" );
        PrecisionRecallM prEuclidean( "Euclidean" );
        PrecisionRecallM prRand( "RANDOM" );

        Euclidean euclidean( db );

        std::cout << "eval start....??" << std::endl;
        experiment4_evaluate( db,
                              dust,
                              topIdDists ,
                              pr,
                              prEuclidean,
                              prRand
        );
        std::cout << "eval finish!!!!" << std::endl;
        
        fout << -1 << " " << "-1" << " "
             << pr.getPrecision() <<  " "
             << pr.getRecall() << " "
             << pr.getF1() << " "
             << pr.getTime() << " "
             << prEuclidean.getPrecision() << " "
             << prEuclidean.getRecall() << " "
             << prEuclidean.getF1() << " "
             << prEuclidean.getTime() << " "
             << prRand.getPrecision() << " "
             << prRand.getRecall() << " "
             << prRand.getF1() << " "
             << prRand.getTime() << " " << std::endl;

        prEuclidean.print();
        pr.print();

        fout.flush();
        fout.close();
    }
}

void generateFDustTables( int argc, char **argv ){
    TimeSeriesCollection db;
    DUST dust( db );
    dust.buildFDustTables(o.wfile);
}

