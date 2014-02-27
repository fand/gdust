#include "ckernel.hpp"
#include "randomvariable.hpp"
#include "config.hpp"

#include <cmath>
#include <cstdlib>

//using namespace std;

double
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

double
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


double
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


// calculate p(y|r(y)=v)p(r(y)=v)
double
c_f1 (double v, double *params)
{
    double p1 = c_myPDF( params[ PARAM_X_DISTRIBUTION ],   // distribution
                         0.0,                                // mean
                         params[ PARAM_X_STDDEV ],         // stddev
                         params[ PARAM_X_OBSERVATION ]-v ); // target

    double p2 = c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}


// calculate p(y|r(y)=v)p(r(y)=v)
double
c_f2 (double v, double *params)
{
    double p1 = c_myPDF( params[ PARAM_Y_DISTRIBUTION ],   // distribution
                         0.0,                                // mean
                         params[ PARAM_Y_STDDEV ],         // stddev
                         params[ PARAM_Y_OBSERVATION ] - v );  // target

    double p2 = c_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}


// p(r(x)=z|x) * p(r(y)=z|y)
double
c_f3 (double z, double *params)
{
    int    x_dist   = (int)params[ PARAM_X_DISTRIBUTION ];
    double x        =      params[ PARAM_X_OBSERVATION ] - 0.1;
    double x_stddev =      params[ PARAM_X_STDDEV ];
    int    y_dist   = (int)params[ PARAM_Y_DISTRIBUTION ];
    double y        =      params[ PARAM_Y_OBSERVATION ] + 0.1;
    double y_stddev =      params[ PARAM_Y_STDDEV ];

    double p1, p2;

    if (x_dist == RANDVAR_UNIFORM) {
        double
            x_adjust = 0;
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
c_f4 (double k, double *params)
{
    return 1.0;
}

double myrand(){
    return ((double) rand() / (RAND_MAX));
}


double
c_dust_kernel (double *params, int time)
{
    double in1, in2, in3;
    double o1 = 0.0;
    double o2 = 0.0;
    double o3 = 0.0;

    // put (f1, f2, f3) into (o1, o2, o3) for all samples
    // Get sum of (o1, o2, o3) for all threads
    for (int i = 0; i < INTEGRATION_SAMPLES; i++) {
        in1 = myrand() * RANGE_WIDTH + RANGE_MIN;
        in2 = myrand() * RANGE_WIDTH + RANGE_MIN;
        in3 = myrand() * RANGE_WIDTH + RANGE_MIN;
        o1 += c_f1( in1, params );
        o2 += c_f2( in2, params );
        o3 += c_f3( in3, params );
    }

    double r = (double)RANGE_WIDTH / INTEGRATION_SAMPLES;
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


