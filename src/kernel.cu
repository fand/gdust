#include "kernel.hpp"
#include "randomvariable.hpp"


#define PI_FLOAT 3.14159265358979323846264338327950288f
#define VERYSMALL 1E-8
#define SQRT3 1.73205081f
#define RANGE_VALUE SQRT3*10

#define INTEG_RANGE_MAX 16
#define INTEG_RANGE_MIN -16

#define PARAM_SIZE 6
#define PARAM_X_DISTRIBUTION 0
#define PARAM_X_OBSERVATION 1
#define PARAM_X_STDDEV 2
#define PARAM_Y_DISTRIBUTION 3
#define PARAM_Y_OBSERVATION 4
#define PARAM_Y_STDDEV 5


__device__ bool
check_uniform_lower (float lower, float *result)
{
    if (isfinite(lower)) {
        return true;
    } else {
        *result = nan("");
        return false;
    }
}


__device__ bool
check_uniform_upper (float upper, float *result)
{
    if (isfinite(upper)) {
        return true;
    }
    else {
        *result = nan("");
        return false;
    }
}


__device__ bool
check_uniform (float lower, float upper, float *result)
{
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
check_uniform_x (float const& x, float *result)
{
    if (isfinite(x)) {
        return true;
    } else {
        *result = nan("");
        return false;
    }
}
    

__device__ bool
check_location (float location, float * result)
{
    if (!(isfinite(location))) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
check_x (float x, float *result)
{
    if(!(isfinite(x))) {
        *result = nan("");
        return false;
    }
    return true;
}


__device__ bool
check_scale(float scale, float *result)
{
    if ((scale <= 0) || !(isfinite(scale))) {
        *result = nanf("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
verify_lambda (float l, float *result)
{
    if (l <= 0) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}


__device__ bool
verify_exp_x (float x, float *result)
{
    if (x < 0) {
        *result = nan("");
        return false;
    } else {
        return true;
    }
}





__device__ float
pdf_uniform (float lower, float upper, float x)
{
    if ((x < lower) || (x > upper)) {
        return 0.0f;
    }

    return 1.0f / (upper - lower);
}

__device__ float
pdf_normal (float mean, float sd, float x)
{
    if (isinf(x)) { return 0; }  // pdf(infinity) is zero.

    float result = 0.0f;

    float exponent = x - mean;
    exponent *= ( (-1) * exponent );
    exponent /= ( 2 * sd * sd );

    result = __expf( exponent );
    result /= sd * sqrt( 2 * PI_FLOAT );

    return result;
}


__device__ float
myPDF (int distribution, float mean, float stddev, float v)
{
    float ret = -1.0f;
    if (stddev == 0.0f) stddev = 0.2f;

    if (distribution == RANDVAR_UNIFORM) {
        float b = SQRT3 * stddev;
        ret = pdf_uniform( -b, b, v );        
    }
    else if (distribution == RANDVAR_NORMAL) {
        ret = pdf_normal( 0, 1, v / stddev );
    }

    return ret;
}


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
f1 (float v, float *params)
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
f2 (float v, float *params)
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
f3 (float z, float *params)
{
    int x_dist = (int)params[ PARAM_X_DISTRIBUTION ];
    float x = params[ PARAM_X_OBSERVATION ] - 0.1f;
    float x_stddev = params[ PARAM_X_STDDEV ];
    int y_dist = (int)params[ PARAM_Y_DISTRIBUTION ];
    float y = params[ PARAM_Y_OBSERVATION ] + 0.1f;
    float y_stddev = params[ PARAM_Y_STDDEV ];
    
    float p1, p2;

    if (x_dist == RANDVAR_UNIFORM) {
        float x_adjust = 0;
        float y_adjust = 0;

        if (abs(x-z) > x_stddev * SQRT3) {
            x_adjust = myPDF( x_dist, 0, x_stddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - x_stddev * SQRT3 ) ) );
        }
        
        if (abs(y-z) > y_stddev * SQRT3) {
            y_adjust = myPDF( y_dist, 0, y_stddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - y_stddev * SQRT3 ) ) );
        }

        float pdf_x = myPDF( x_dist, 0.0f, x_stddev, x-z ) + x_adjust;
        float pdf_y = myPDF( y_dist, 0.0f, y_stddev, y-z ) + y_adjust;

        p1 = pdf_x * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        p2 = pdf_y * pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
    }
    else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = ( myPDF( x_dist, 0, x_stddev, x-z ) *
               pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
        p2 = ( myPDF( y_dist, 0, y_stddev, y-z ) *
               pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
    }

    return p1 * p2;
}


__device__ float
f4 (float k, float *params)
{
    return 1.0f;
}


// With seq on global memory
__global__ void
distance_kernel (float *seq_GPU,
                 float *samples_GPU,
                 float *dust_GPU)
{
    float *p_param = seq_GPU  + blockIdx.x * PARAM_SIZE;    
    float *p_dust  = dust_GPU + blockIdx.x;

    dust_kernel(p_param, samples_GPU, p_dust);
}


__device__ void
dust_kernel (float *params,
             float *in,
             float *answer_GPU)
{
    float in1, in2, in3;
    int offset1 = blockIdx.x * INTEGRATION_SAMPLES;
    int offset2 = offset1 + INTEGRATION_SAMPLES * gridDim.x;
    int offset3 = offset2 + INTEGRATION_SAMPLES * gridDim.x;
    
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;

    __shared__ float sdata1[TPB];
    __shared__ float sdata2[TPB];
    __shared__ float sdata3[TPB];

    // MAP PHASE
    // put (f1, f2, f3) into (o1, o2, o3) for all samples
    for (int i = threadIdx.x; i < INTEGRATION_SAMPLES; i += blockDim.x) {
        in1 = in[i + offset1] * RANGE_WIDTH + RANGE_MIN;
        in2 = in[i + offset2] * RANGE_WIDTH + RANGE_MIN;
        in3 = in[i + offset3] * RANGE_WIDTH + RANGE_MIN;        
        o1 += f1( in1, params );
        o2 += f2( in2, params );
        o3 += f3( in3, params );
    }
    
    // REDUCE PHASE
    // Get sum of (o1, o2, o3) for all threads
    sdata1[threadIdx.x] = o1;
    sdata2[threadIdx.x] = o2;
    sdata3[threadIdx.x] = o3;
    reduceBlock<TPB>(sdata1, sdata2, sdata3);

    float r = (float)RANGE_WIDTH / INTEGRATION_SAMPLES;

    if (threadIdx.x == 0) {
        float d = -log10(sdata3[0] / (sdata1[0] * sdata2[0] * r));
        if (d < 0) d = 0.0f;
        *answer_GPU = d;
    }
}



template<unsigned int blockSize>
__device__ void
reduceBlock (float *sdata1, float *sdata2, float *sdata3)
{
    // make sure all threads are ready
    __syncthreads();
    
    unsigned int tid = threadIdx.x;
    float mySum1 = sdata1[tid];
    float mySum2 = sdata2[tid];
    float mySum3 = sdata3[tid];    
    
    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 256];
            sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 256];
            sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 256];            
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 128];
            sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 128];
            sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 128];            
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid <  64) {
            sdata1[tid] = mySum1 = mySum1 + sdata1[tid + 64];
            sdata2[tid] = mySum2 = mySum2 + sdata2[tid + 64];
            sdata3[tid] = mySum3 = mySum3 + sdata3[tid + 64];            
        }
        __syncthreads();
    }

    if (tid < 32) {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *smem1 = sdata1;
        volatile float *smem2 = sdata2;
        volatile float *smem3 = sdata3;

        if (blockSize >=  64) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 32];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 32];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 32];
        }

        if (blockSize >=  32) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 16];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 16];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 16];
        }

        if (blockSize >=  16) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 8];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 8];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 8];
        }

        if (blockSize >=   8) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 4];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 4];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 4];
        }

        if (blockSize >=   4) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 2];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 2];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 2];
        }

        if (blockSize >=   2) {
            smem1[tid] = mySum1 = mySum1 + smem1[tid + 1];
            smem2[tid] = mySum2 = mySum2 + smem2[tid + 1];
            smem3[tid] = mySum3 = mySum3 + smem3[tid + 1];
        }
    }
}

