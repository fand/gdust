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

    float exponent = x - mean;
    exponent *= ( (-1) * exponent );
    exponent /= ( 2 * sd * sd );

    result = __expf( exponent );
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
f1 ( float v, float *params )
{
    float p1 = myPDF( params[ PARAM_X_DISTRIBUTION ],   // distribution
                      0.0f,                                // mean
                      params[ PARAM_X_STDDEV ],         // stddev
                      params[ PARAM_X_OBSERVATION ]-v ); // target
    
    float p2 = pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
f2 ( float v, float *params )
{
    float p1 = myPDF( params[ PARAM_Y_DISTRIBUTION ],   // distribution
                      0.0f,                                // mean
                      params[ PARAM_Y_STDDEV ],         // stddev
                      params[ PARAM_Y_OBSERVATION ] - v );  // target
    
    float p2 = pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );
    
    return p1 * p2;
}


// p(r(x)=z|x) * p(r(y)=z|y)
__device__ float
f3 ( float z, float *params )
{
    int x_dist = (int)params[ PARAM_X_DISTRIBUTION ];
    float x = params[ PARAM_X_OBSERVATION ] - 0.1f;
    float x_stddev = params[ PARAM_X_STDDEV ];
    int y_dist = (int)params[ PARAM_Y_DISTRIBUTION ];
    float y = params[ PARAM_Y_OBSERVATION ] + 0.1f;
    float y_stddev = params[ PARAM_Y_STDDEV ];

    
    float p1, p2;

    if ( x_dist == RANDVAR_UNIFORM ) {
        float x_adjust = 0;
        float y_adjust = 0;

        if ( abs(x-z) > x_stddev * SQRT3 ) {
            x_adjust = myPDF( x_dist, 0, x_stddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - x_stddev * SQRT3 ) ) );
        }
        
        if ( abs(y-z) > y_stddev * SQRT3 ) {
            y_adjust = myPDF( y_dist, 0, y_stddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - y_stddev * SQRT3 ) ) );
        }

        float pdf_x = myPDF( x_dist, 0.0f, x_stddev, x-z ) + x_adjust;
        float pdf_y = myPDF( y_dist, 0.0f, y_stddev, y-z ) + y_adjust;

        p1 = pdf_x * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        p2 = pdf_y * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        
    } else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = ( myPDF( x_dist, 0, x_stddev, x-z ) *
               pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
        p2 = ( myPDF( y_dist, 0, y_stddev, y-z ) *
               pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
    }

    return p1 * p2;
}


__device__ float
f4 ( float k, float *params )
{
    return 1.0;
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
    
    // random input "input_array[idx]" is (0, 1]
    // so we need to apply them to the range of integration.
     float input = ( input_array[ idx ] * (range_max - range_min) ) + range_min;
    
    if (fnum == 1) output_array[ idx ] = f1( input, (float *)params );
    if (fnum == 2) output_array[ idx ] = f2( input, (float *)params );
    if (fnum == 3) output_array[ idx ] = f3( input, (float *)params );
    if (fnum == 4) output_array[ idx ] = f4( input, (float *)params );
}





__global__ void phi_kernel(
    void *params,
    float *in,
    float *out1,
    float *out2,
    float *out3,
    float range_min,
    float range_max,
    int loop_num)
{
    float input;
    
    float o1 = 0;
    float o2 = 0;
    float o3 = 0;


    // MAP PHASE
    int i, index;
    for (i = 0; i < loop_num; i++) {
        input = (in[loop_num * i + threadIdx.x] * (range_max - range_min)) + range_min;
        
        o1 += f1( input, (float *)params );
        o2 += f2( input, (float *)params );
        o3 += f3( input, (float *)params );
    }

    out1[threadIdx.x] = o1;
    out2[threadIdx.x] = o2;
    out3[threadIdx.x] = o3;



    // REDUCE PHASE
    
    
    
    
}
