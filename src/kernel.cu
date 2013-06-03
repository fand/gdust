#include "kernel.hpp"
#include "randomvariable.hpp"


#define PI_FLOAT 3.14159265358979323846264338327950288f
#define VERYSMALL 1E-8
#define SQRT3 1.73205081f
#define RANGE_VALUE SQRT3*10

#define INTEG_RANGE_MAX 16
#define INTEG_RANGE_MIN -16

#define PARAM_INT1 0
#define PARAM_INT2 1
#define PARAM_X_DISTRIBUTION 2
#define PARAM_X_OBSERVATION 3
#define PARAM_X_STDDEV 4
#define PARAM_Y_DISTRIBUTION 5
#define PARAM_Y_OBSERVATION 6
#define PARAM_Y_STDDEV 7



__device__ bool
check_uniform_lower(float lower, float *result) {
    if (isfinite(lower)) {
        return true;
    } else {
        *result = nan("");
        return false;
    }
}


__device__ bool
check_uniform_upper(float upper, float *result) {
    if (isfinite(upper)){
        return true;
    } else {
        *result = nan("");
        return false;
    }
}


__device__ bool
check_uniform(float lower, float upper, float *result) {
    if (check_uniform_lower(lower, result) == false) { return false; }
    else if (check_uniform_upper(upper, result) == false) { return false; }
    // If lower == upper then 1 / (upper-lower) = 1/0 = +infinity!
    else if (lower >= upper) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
check_uniform_x(float const& x, float *result) {
    if (isfinite(x)) {
        return true;
    } else {
        *result = nan("");
        return false;
    }
}
    

__device__ bool
check_location(float location, float * result) {
    if (!(isfinite(location))) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
check_x(float x, float *result) {
    if(!(isfinite(x))) {
        *result = nan("");
        return false;
    }
    return true;
}


__device__ bool
check_scale(float scale, float *result) {
    if((scale <= 0) || !(isfinite(scale))) {
        *result = nanf("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
verify_lambda(float l, float *result) {
    if (l <= 0) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
verify_exp_x(float x, float *result) {
    if (x < 0) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}





__device__ float pdf_uniform( float lower, float upper, float x )
{
    if ((x < lower) || (x > upper)) {
        return 0;
    }

    return 1.0 / (upper - lower);
}

__device__ float pdf_normal( float mean, float sd, float x )
{
    if(isinf(x)) return 0; // pdf(infinity) is zero.

    float result = 0.0f;
    // if(false == check_scale(sd, &result)) return result;
    // if(false == check_location(mean, &result)) return result;
    // if(false == check_x(x, &result)) return result;

    float exponent = x - mean;
    exponent *= ( (-1) * exponent );
    exponent /= ( 2 * sd * sd );

    result = exp( exponent );
    result /= sd * sqrt( 2 * PI_FLOAT );

    return result;
}


__device__ float
myPDF( int distribution, float mean, float stddev, float v )
{
    float ret = -1.0f;
    if(stddev == 0.0f) stddev = 0.2f;

    if ( distribution == RANDVAR_UNIFORM )
    {
        float b = SQRT3 * stddev;
        ret = pdf_uniform( -b, b, v );
        
    } else if ( distribution == RANDVAR_NORMAL )
    {
        ret = pdf_normal( 0, 1, v / stddev );
    }

    return ret;
}


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
f1 ( float k, size_t dim, float *fp )
{
    float x = fp[ PARAM_X_OBSERVATION ];
    float v = k;  // random input

    float p1 = myPDF( fp[ PARAM_X_DISTRIBUTION ],    // distribution
                      0.0f,                          // mean
                      fp[ PARAM_X_STDDEV ],          // stddev
                      x-v );                         // target
    
    float p2 = pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

//    if (isnan(p1) || isnan(p2)) return 89898989;
    return p1 * p2;
}


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
f2 ( float k, size_t dim, float *fp )
{
    float y = fp[ PARAM_Y_OBSERVATION ];
    float v = k;

    float p1 = myPDF( fp[ PARAM_Y_DISTRIBUTION ], // distribution
                      0,                          // mean
                      fp[ PARAM_Y_STDDEV ],       // stddev
                      y-v );                      // target
    
    float p2 = pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );
    
    return p1 * p2;
}


// p(r(x)=z|x) * p(r(y)=z|y)
__device__ float
f3 ( float k, size_t dim, float *fp )
{
    float int1 = fp[ PARAM_INT1 ];
    float int2 = fp[ PARAM_INT2 ];
    int xdistrib = (int)fp[ PARAM_X_DISTRIBUTION ];
    float x = fp[ PARAM_X_OBSERVATION ] - 0.1f;
    float xstddev = fp[ PARAM_X_STDDEV ];
    int ydistrib = (int)fp[ PARAM_Y_DISTRIBUTION ];
    float y = fp[ PARAM_Y_OBSERVATION ] + 0.1f;
    float ystddev = fp[ PARAM_Y_STDDEV ];
    float z = k;

//    assert( xstddev >= 0 );
//    assert( ystddev >= 0 );
//    assert( ( int1 != 0 ) && ( int2 != 0 ) );

    float p1, p2;

    if ( xdistrib == RANDVAR_UNIFORM ) {
        float xadjust = 0;
        float yadjust = 0;

        if ( abs(x-z) > xstddev * SQRT3 ) {
            xadjust = myPDF( xdistrib, 0, xstddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - xstddev * SQRT3 ) ) );
        }
        
        if ( abs(y-z) > ystddev * SQRT3 ) {
            yadjust = myPDF( ydistrib, 0, ystddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - ystddev * SQRT3 ) ) );
        }

        float pdfx = myPDF( xdistrib, 0, xstddev, x-z ) + xadjust;
        float pdfy = myPDF( ydistrib, 0, ystddev, y-z ) + yadjust;

        p1 = pdfx * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) / int1;
        p2 = pdfy * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) / int2;
        
    } else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = myPDF( xdistrib, 0, xstddev, x-z ) * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) / int1;
        p2 = myPDF( ydistrib, 0, ystddev, y-z ) * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) / int2;
    }

    float ret = p1 * p2;

    return ret;
}


__device__ float
f4 ( float k, size_t dim, float *fp )
{
    float ret = pdf_uniform( -1, 1, k );
    return ret;
}



__global__ void integrate_kernel(
    void *params,
    int fnum,
    float *input_array,
    float *output_array,
    float range_min,
    float range_max )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // random input "input_array[idx]" is 0 to 1,
    // so we need to apply them to the range of integration.
    float input = ( input_array[ idx ] * (range_max - range_min) ) + range_min;
    
    if (fnum == 1) output_array[ idx ] = f1( input, 1, (float *)params );
    if (fnum == 2) output_array[ idx ] = f2( input, 1, (float *)params );
    if (fnum == 3) output_array[ idx ] = f3( input, 1, (float *)params );
    if (fnum == 4) output_array[ idx ] = f4( input, 1, (float *)params );    
}

