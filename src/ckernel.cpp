#include "ckernel.hpp"
#include "RandomVariable.hpp"
#include "config.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <omp.h>


//
// PDF functions
//__________________

inline double
c_pdf_uniform (double lower, double upper, double x)
{
    if ((x < lower) || (x > upper)) {
        return 0.0;
    }
    if (lower == x && upper == x) {
        return 0.0;
    }
    return 1.0 / (upper - lower);
}

inline double
c_pdf_normal (double mean, double sd, double x)
{
    if (isinf(x) || sd <= 0 || isinf(sd) || isinf(mean)) {
        return 0.0;
    }

    double result = 0.0;

    double exponent = x - mean;
    exponent *= -exponent;
    exponent /= (2 * sd * sd);

    result = exp(exponent);
    result /= sd * sqrt(2 * PI_DOUBLE);

    return result;
}

inline double
c_myPDF (int distribution, double mean, double stddev, double v)
{
    double ret = -1.0f;
    if (stddev == 0.0f) stddev = 0.2f;

    if (distribution == RANDVAR_UNIFORM) {
        double b = SQRT3 * stddev;
        ret = c_pdf_uniform( -b, b, v );
    }
    else if (distribution == RANDVAR_NORMAL) {
        ret = c_pdf_normal( 0, 1, v / stddev );
    }

    return ret;
}


//
// Integrand functions in dust
//________________________________

//!
// Calculate p(x|r(x)=v)p(r(x)=v).
//
// @param {float}   v  - Random value
// @param {float[]} xy - An array containing x & y
inline double
c_f1 (double v, double *xy)
{
    double p1 = c_myPDF( xy[ TUPLE_X_DISTRIBUTION ],     // distribution
                         0.0,                            // mean
                         xy[ TUPLE_X_STDDEV ],           // stddev
                         xy[ TUPLE_X_OBSERVATION ]-v );  // target

    double p2 = c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}

//!
// Calculate p(y|r(y)=v)p(r(y)=v).
// Almost same as c_f1.
//
inline double
c_f2 (double v, double *xy)
{
    double p1 = c_myPDF( xy[ TUPLE_Y_DISTRIBUTION ],       // distribution
                         0.0,                              // mean
                         xy[ TUPLE_Y_STDDEV ],             // stddev
                         xy[ TUPLE_Y_OBSERVATION ] - v );  // target

    double p2 = c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}

//!
// Calculate p(r(x)=z|x) * p(r(y)=z|y).
//
// @param {float}   z  - Random value
// @param {float[]} xy - An array containing x & y
//
inline double
c_f3 (double z, double *xy)
{
    int    x_dist   = (int)xy[ TUPLE_X_DISTRIBUTION ];
    double x        =      xy[ TUPLE_X_OBSERVATION ] - 0.1;
    double x_stddev =      xy[ TUPLE_X_STDDEV ];
    int    y_dist   = (int)xy[ TUPLE_Y_DISTRIBUTION ];
    double y        =      xy[ TUPLE_Y_OBSERVATION ] + 0.1;
    double y_stddev =      xy[ TUPLE_Y_STDDEV ];

    double p1, p2;

    if (x_dist == RANDVAR_UNIFORM) {
        double x_adjust = 0;
        double y_adjust = 0;

        if (abs(x-z) > x_stddev * SQRT3) {
            x_adjust = c_myPDF( x_dist, 0, x_stddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - x_stddev * SQRT3 ) ) );
        }

        if (abs(y-z) > y_stddev * SQRT3) {
            y_adjust = c_myPDF( y_dist, 0, y_stddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - y_stddev * SQRT3 ) ) );
        }

        double pdf_x = c_myPDF( x_dist, 0.0, x_stddev, x-z ) + x_adjust;
        double pdf_y = c_myPDF( y_dist, 0.0, y_stddev, y-z ) + y_adjust;

        p1 = pdf_x * c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        p2 = pdf_y * c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
    }
    else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = ( c_myPDF( x_dist, 0, x_stddev, x-z ) *
               c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
        p2 = ( c_myPDF( y_dist, 0, y_stddev, y-z ) *
               c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
    }

    return p1 * p2;
}

double
c_f4 (double k, double *xy)
{
    return 1.0;
}


//
// Functions for dust / DUST
//______________________________

double
c_dust_kernel (double *xy, double *samples, int time)
{
    double o1 = 0.0;
    double o2 = 0.0;
    double o3 = 0.0;

    int offset = time * 3 * INTEGRATION_SAMPLES;
    double *local_samples = samples + offset;
    for (int i = 0; i < INTEGRATION_SAMPLES; ++i) {
        o1 += c_f1(local_samples[i * 3    ], xy);
        o2 += c_f2(local_samples[i * 3 + 1], xy);
        o3 += c_f3(local_samples[i * 3 + 2], xy);
    }

    double r = (double) RANGE_WIDTH / INTEGRATION_SAMPLES;
    double int1 = o1 * r;
    double int2 = o2 * r;
    double int3 = o3 * r;
    if (int1 < VERYSMALL) int1 = VERYSMALL;
    if (int2 < VERYSMALL) int2 = VERYSMALL;
    if (int3 < 0.0) int3 = 0.0;

    double d = -log10(int3 / (int1 * int2));

    if (d < 0.0) { d = 0.0; }

    return d;
}
