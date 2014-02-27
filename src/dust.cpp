#include "dust.hpp"
#include "common.hpp"
#include "config.hpp"

#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

#include <cassert>

#include "ckernel.hpp"

inline double
clean_probability (double p)
{
    return (p < 0.0) ? 0.0 : p;
}


double
DUST::c_distance (TimeSeries &ts1, TimeSeries &ts2, int n)
{
    double dist = 0;

    for (int i=0; i < n; i++) {
        RandomVariable x = ts1.at(i);
        RandomVariable y = ts2.at(i);

        double params[] = {
            x.distribution,
            x.observation,
            x.stddev,
            y.distribution,
            y.observation,
            y.stddev
        };

        dist += c_dust_kernel(params, i);
    }

    return sqrt(dist);
}


void
DUST::init()
{
    T = gsl_rng_default;
    r_rng = gsl_rng_alloc (T);
    lookupTablesAvailable = false;
}


double
DUST::phi(RandomVariable &x, RandomVariable &y)
{
    // RandomVariable *pair[2];
    // pair[0] = &x;
    // pair[1] = &y;

    // double int1 = clean_probability( this->integrate(c_f1, (void *)pair) );
    // double int2 = clean_probability( this->integrate(c_f2, (void *)pair) );

    // if (int1 == 0) { int1 = VERYSMALL; }
    // if (int2 == 0) { int2 = VERYSMALL; }

    // assert( int1 <= 1 );
    // assert( int2 <= 1 );
    // assert( int1 != 0 && int2 != 0 );

    // double params[8] = {
    //     int1,
    //     int2,
    //     x.distribution,
    //     x.observation,
    //     x.stddev,
    //     y.distribution,
    //     y.observation,
    //     y.stddev
    // };

    // double int3 = integrate( c_f3, ( void * )params );
    // int3 = clean_probability( int3 );

    // if (int3 == 0) {
    //     std::cerr << "dust:" << std::endl;
    //     std::cerr << "int1:" << int1 << " int2:" << int2 << " int3:" << int3 << std::endl;
    //     std::cerr << "x : " << x.distribution << " : " << x.observation << " : " << x.stddev << std::endl;
    //     std::cerr << "y : " << y.distribution << " : " << y.observation << " : " << y.stddev << std::endl;
    //     // exit(0);
    // }

    // assert( NOTNANINF(int3) );
    // return int3;
    return 0;
}


DUST::~DUST()
{
    gsl_rng_free( r_rng );
}


DUST::DUST(TimeSeriesCollection &collection, const char *lookUpTablesPath)
{
    this->collection = collection;
    this->init();

    if (lookUpTablesPath) {
        readLookUpTables(lookUpTablesPath);
    }
}


std::vector<int>
DUST::rangeQuery( TimeSeries ts, double threshold )
{
    std::vector< int > ids;

    for (unsigned int i = 0; i < collection.sequences.size(); i++) {
        std::cout << "rangeQuery loop " << i << ", length " << ts.length() << std::endl;

        double dist = 0;
        for (unsigned int j = 0; j < ts.length(); j++) {
            RandomVariable a = ts.at(j);
            RandomVariable b = collection.sequences[i].at(j);
            dist += pow( dust( a, b ), 2 );

            if (dist > threshold * threshold) {
                break;
            }
        }

        if (sqrt(dist) <= threshold) {
            ids.push_back( i+1 );
        }
    }

    return ids;
}


double
DUST::distance(TimeSeries &ts1, TimeSeries &ts2, int n)
{
    int lim;
    if (n == -1) {
        lim = fmin(ts1.length(), ts2.length());
    }
    else {
        lim = fmin(n, fmin(ts1.length(), ts2.length()));
    }
    return this->c_distance(ts1, ts2, lim);
}


void
DUST::readLookUpTables(const char *lookUpTablesPath)
{
    std::ifstream fin( lookUpTablesPath );

    if (fin == NULL) {
        FATAL("unable to read lookuptables from file");
    }

    std::string line;

    for (int i = 0; i < 3; i++) {
        for (int j0 = 0; j0 < STDDEV_STEPS; j0++) {
            for (int j1 = 0; j1 < STDDEV_STEPS; j1++) {
                for (int k = 0; k < DISTANCE_STEPS; k++) {
                    lookuptables[ i ][ j0 ][ j1 ][ k ] = -1;
                }
            }
        }
    }

    while (getline( fin, line )) {
        std::stringstream lineStream( line );

        int distribution;
        double stddev0,stddev1, distance, dustdistance;

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

    for (int i = 0; i < 3; i++) {
        for (int j0 = 0; j0 < STDDEV_STEPS; j0++) {
            for (int j1 = 0; j1 < STDDEV_STEPS; j1++) {
                for (int k = 0; k < DISTANCE_STEPS; k++) {
                    assert(
                        lookuptables[ i ][ j0 ][ j1 ][ k ] >= 0
                    ); // this ensures that all needed values are present.
                }
            }
        }
    }

    lookupTablesAvailable = true;
}


double
DUST::dust( RandomVariable &x, RandomVariable &y )
{
    if (lookupTablesAvailable) {
        assert( x.distribution == y.distribution );

        double distance =
            ceil( ABS(x.observation - y.observation) * (1 / DISTANCE_STEP) ) /
            (1 / DISTANCE_STEP );

        if (distance > DISTANCE_END) {
            distance = DISTANCE_END;
        }

        assert( x.distribution >= 1 && x.distribution <= 3 );
        assert( x.stddev >= STDDEV_BEGIN && x.stddev <= STDDEV_END );
        assert( distance >= DISTANCE_BEGIN && distance <= DISTANCE_END );

        int stddev0_offset = (int)round( x.stddev / STDDEV_STEP );
        int stddev1_offset = (int)round( y.stddev / STDDEV_STEP );
        int distance_offset = (int)( distance / DISTANCE_STEP );

        assert( stddev0_offset >= 0 && stddev0_offset < STDDEV_STEPS );
        assert( stddev1_offset >= 0 && stddev1_offset < STDDEV_STEPS );
        assert( distance_offset >= 0 && distance_offset < DISTANCE_STEPS );

        double dustdist = lookuptables[ x.distribution - 1 ][ stddev0_offset ][ stddev1_offset ][ distance_offset ];

        if (dustdist < 0) {
            std::cerr << "dustdist negative: " << dustdist << " " << x.distribution << std::endl;
        }

        assert( dustdist >= 0 );
        return dustdist;
    }

    double K = 0; // K disabled. we do not need normalization. MAX(-log10(phi(x,x)), -log10(phi(y,y)));
    // if(K < 0) K = 0;
    // assert( NOTNANINF(K) );

    double distance = -log10(phi(x,y)) - K;
    if (distance < 0) {
        distance = 0;
    }
    distance = sqrt(distance);
    assert( NOTNANINF(distance) );

    return distance;
}


double
DUST::dtw(TimeSeries &ts1, TimeSeries &ts2)
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


void
DUST::calcCost(TimeSeries &ts1, TimeSeries &ts2, double *table_d, double *table_g, int len1, int len2)
{
    for (int i = 0; i < len1; i++) {
        RandomVariable r1 = ts1.at(i);

        for (int j = 0; j < len2; j++) {
            RandomVariable r2 = ts2.at(j);
//            table_d[ i*len2 + j ] = pow( dust( r1, r2 ), 2 );
            table_d[ i*len2 + j ] = dust( r1, r2 );
        }
    }
}


void
DUST::calcGamma(double *table_d, double *table_g, int len1, int len2)
{
    table_g[0] = table_d[0];

    for (int i = 1; i < len1; i++) {
        table_g[ i*len2 ] = table_d[ i*len2 ] + table_g[ (i-1) * len2 ];
    }

    for (int i = 1; i < len2; i++) {
        table_g[i] = table_d[i] + table_g[i-1];
    }

    for (int i = 1; i< len1; i++) {
        for (int j = 1; j < len2; j++) {
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


double
DUST::calcSum(double *table_d, double *table_g, int len1, int len2)
{
    double sum = 0.0;

    int i = len1 - 1;
    int j = len2 - 1;

    while (i > 0 || j > 0) {

        sum += table_g[ i*len2 + j ];

        if (i == 0) {
            j--;
        }
        else if (j == 0) {
            i--;
        }
        else {
            double m = std::min(
                std::min(
                    table_g[ (i-1)*len2 + j ],
                    table_g[ i*len2 + j-1 ] ),
                table_g[ (i-1)*len2 + j-1 ]
            );
            if (m == table_g[ (i-1)*len2 + j ]) {
                i--;
            }
            else if (m == table_g[ i*len2 + j-1 ]) {
                j--;
            }
            else {
                i--; j--;
            }
        }
    }

    return sum;
}


void
DUST::buildFDustTables( const char *path )
{
    std::ofstream fout( path );

    if (!(fout)) {
        FATAL( "error opening outputfile." );
    }

    fout.precision( 4 );

    for (int distribution = 1; distribution <= 3; distribution++) {
        for (double stddev0 = STDDEV_BEGIN; stddev0 < STDDEV_END+STDDEV_STEP/2.0; stddev0+=STDDEV_STEP) {
            for (double stddev1 = stddev0; stddev1 < STDDEV_END+STDDEV_STEP/2.0; stddev1+=STDDEV_STEP) {
                for(double distance = DISTANCE_BEGIN; distance < DISTANCE_END+DISTANCE_STEP/2.0; distance+= DISTANCE_STEP) {

                    RandomVariable x = RandomVariable( distribution, (float)0, (float)0, (float)stddev0 );
                    RandomVariable y = RandomVariable( distribution,
                                                       (float)distance,
                                                       (float)distance,
                                                       (float)stddev1 );
                    double d = dust(x, y);
                    std::cerr << "distribution:" << distribution << " stddev0:" << stddev0 << " stddev1:" << stddev1 << " distance:" << distance << " dustdistance:" << d << std::endl;

                    fout << distribution << " " << stddev0 << " " << stddev1 <<  " " << distance << " " << d << "\n";
                    fout << distribution << " " << stddev1 << " " << stddev0 <<  " " << distance << " " << d << "\n";
                }
            }
        }
    }
}

