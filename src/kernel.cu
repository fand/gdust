#include "kernel.hpp"
#include "randomvariable.hpp"
#include "config.hpp"
#include <stdio.h>


//
// PDF functions
//__________________

__device__ float
g_pdf_uniform (float lower, float upper, float x)
{
    if ((x < lower) || (x > upper)) {
        return 0.0f;
    }
    if (lower == x && upper == x) {
        return 0.0f;
    }
    return 1.0f / (upper - lower);
}

__device__ float
g_pdf_normal (float mean, float sd, float x)
{
    if (isinf(x) || sd <= 0 || isinf(sd) || isinf(mean)) {
        return 0.0f;
    }

    float result = 0.0f;

    float exponent = x - mean;
    exponent *= -exponent;
    exponent /= (2 * sd * sd);

    result = __expf(exponent);
    result /= sd * sqrt(2 * PI_FLOAT);

    return result;
}

__device__ float
g_myPDF (int distribution, float mean, float stddev, float v)
{
    float ret = -1.0f;
    if (stddev == 0.0f) stddev = 0.2f;

    if (distribution == RANDVAR_UNIFORM) {
        float b = SQRT3 * stddev;
        ret = g_pdf_uniform( -b, b, v );
    }
    else if (distribution == RANDVAR_NORMAL) {
        ret = g_pdf_normal( 0, 1, v / stddev );
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
__device__ float
g_f1 (float v, float *xy)
{
    float p1 = g_myPDF( xy[ TUPLE_X_DISTRIBUTION ],     // distribution
                        0.0f,                           // mean
                        xy[ TUPLE_X_STDDEV ],           // stddev
                        xy[ TUPLE_X_OBSERVATION ]-v );  // target

    float p2 = g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}

//!
// Calculate p(y|r(y)=v)p(r(y)=v).
// Almost same as g_f1.
//
__device__ float
g_f2 (float v, float *xy)
{
    float p1 = g_myPDF( xy[ TUPLE_Y_DISTRIBUTION ],       // distribution
                        0.0f,                             // mean
                        xy[ TUPLE_Y_STDDEV ],             // stddev
                        xy[ TUPLE_Y_OBSERVATION ] - v );  // target

    float p2 = g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}

//!
// Calculate p(r(x)=z|x) * p(r(y)=z|y).
//
// @param {float}   z  - Random value
// @param {float[]} xy - An array containing x & y
//
__device__ float
g_f3 (float z, float *xy)
{
    int   x_dist   = (int)xy[ TUPLE_X_DISTRIBUTION ];
    float x        =      xy[ TUPLE_X_OBSERVATION ] - 0.1f;
    float x_stddev =      xy[ TUPLE_X_STDDEV ];
    int   y_dist   = (int)xy[ TUPLE_Y_DISTRIBUTION ];
    float y        =      xy[ TUPLE_Y_OBSERVATION ] + 0.1f;
    float y_stddev =      xy[ TUPLE_Y_STDDEV ];

    float p1, p2;

    if (x_dist == RANDVAR_UNIFORM) {
        float x_adjust = 0;
        float y_adjust = 0;

        if (abs(x-z) > x_stddev * SQRT3) {
            x_adjust = g_myPDF( x_dist, 0, x_stddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - x_stddev * SQRT3 ) ) );
        }

        if (abs(y-z) > y_stddev * SQRT3) {
            y_adjust = g_myPDF( y_dist, 0, y_stddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - y_stddev * SQRT3 ) ) );
        }

        float pdf_x = g_myPDF( x_dist, 0.0f, x_stddev, x-z ) + x_adjust;
        float pdf_y = g_myPDF( y_dist, 0.0f, y_stddev, y-z ) + y_adjust;

        p1 = pdf_x * g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        p2 = pdf_y * g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
    }
    else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = ( g_myPDF( x_dist, 0, x_stddev, x-z ) *
               g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
        p2 = ( g_myPDF( y_dist, 0, y_stddev, y-z ) *
               g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
    }

    return p1 * p2;
}

__device__ float
g_f4 (float k, float *xy)
{
    return 1.0f;
}

// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
g_f12_multi (float v, float *x)
{
    float p1 = g_myPDF( x[0], 0.0f, x[2], x[1] - v );
    float p2 = g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );
    return p1 * p2;
}

// p(r(x)=z|x) * p(r(y)=z|y)
__device__ float
g_f3_multi (float z, float *x_, float *y_)
{
    int   x_dist   = (int)x_[0];
    float x        =      x_[1] - 0.1f;
    float x_stddev =      x_[2];
    int   y_dist   = (int)y_[0];
    float y        =      y_[1] + 0.1f;
    float y_stddev =      y_[2];

    float p1, p2;

    if (x_dist == RANDVAR_UNIFORM) {
        float x_adjust = 0;
        float y_adjust = 0;

        if (abs(x-z) > x_stddev * SQRT3) {
            x_adjust = g_myPDF( x_dist, 0, x_stddev, 0 ) *
                ( 1 + erf( -( abs(x-z) - x_stddev * SQRT3 ) ) );
        }

        if (abs(y-z) > y_stddev * SQRT3) {
            y_adjust = g_myPDF( y_dist, 0, y_stddev, 0 ) *
                ( 1 + erf( -( abs(y-z) - y_stddev * SQRT3 ) ) );
        }

        float pdf_x = g_myPDF( x_dist, 0.0f, x_stddev, x-z ) + x_adjust;
        float pdf_y = g_myPDF( y_dist, 0.0f, y_stddev, y-z ) + y_adjust;

        p1 = pdf_x * g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
        p2 = pdf_y * g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z );
    }
    else {
        // p(r(x)=z|x) and p(r(y)=z|y)
        p1 = ( g_myPDF( x_dist, 0, x_stddev, x-z ) *
               g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
        p2 = ( g_myPDF( y_dist, 0, y_stddev, y-z ) *
               g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, z ) );
    }

    return p1 * p2;
}

__device__ float
simpson_f1 (float left, float width, float *tuple)
{
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (g_f1(left, tuple) + g_f1(right, tuple) + g_f1(mid, tuple) * 4);
}
__device__ float
simpson_f2 (float left, float width, float *tuple)
{
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (g_f2(left, tuple) + g_f2(right, tuple) + g_f2(mid, tuple) * 4);
}
__device__ float
simpson_f3 (float left, float width, float *tuple)
{
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (g_f3(left, tuple) + g_f3(right, tuple) + g_f3(mid, tuple) * 4);
}
__device__ float
simpson_f12_multi (float left, float width, float *x)
{
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (g_f12_multi(left, x) + g_f12_multi(right, x) + g_f12_multi(mid, x) * 4);
}
__device__ float
simpson_f3_multi (float left, float width, float *x, float *y)
{
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (g_f3_multi(left, x, y) + g_f3_multi(right, x, y) + g_f3_multi(mid, x, y) * 4);
}


//
// Functions for dust / DUST
//______________________________

// With seq on global memory
__global__ void
g_distance_kernel (float *tuples_GPU,
                   float *samples_GPU,
                   float *dust_GPU)
{
    float *tuple = tuples_GPU  + blockIdx.x * TUPLE_SIZE;
    float *dust  = dust_GPU + blockIdx.x;

    g_dust_kernel(tuple, samples_GPU, dust);
}


__device__ void
g_dust_kernel (float *tuple,
               float *samples,
               float *answer_GPU)
{
    float sample1, sample2, sample3;
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
        sample1 = samples[i + offset1] * RANGE_WIDTH + RANGE_MIN;
        sample2 = samples[i + offset2] * RANGE_WIDTH + RANGE_MIN;
        sample3 = samples[i + offset3] * RANGE_WIDTH + RANGE_MIN;
        o1 += g_f1( sample1, tuple );
        o2 += g_f2( sample2, tuple );
        o3 += g_f3( sample3, tuple );
    }

    // REDUCE PHASE
    // Get sum of (o1, o2, o3) for all threads
    sdata1[threadIdx.x] = o1;
    sdata2[threadIdx.x] = o2;
    sdata3[threadIdx.x] = o3;
    g_reduceBlock<TPB>(sdata1, sdata2, sdata3);

    float r = (float)RANGE_WIDTH / INTEGRATION_SAMPLES;

    if (threadIdx.x == 0) {
        float int1 = sdata1[0] * r;
        if (int1 < VERYSMALL) int1 = VERYSMALL;
        float int2 = sdata2[0] * r;
        if (int2 < VERYSMALL) int2 = VERYSMALL;
        float int3 = sdata3[0] * r;
        if (int3 < 0.0f) int3 = 0.0f;

        float dust = -log10(int3 / (int1 * int2));

        if (dust < 0.0) { dust = 0.0f; }
        *answer_GPU = dust;
    }
}

// With seq on global memory
__global__ void
g_distance_simpson_kernel (float *tuples_GPU,
                           float *dust_GPU,
                           int division)
{
    float *tuple = tuples_GPU  + blockIdx.x * TUPLE_SIZE;
    float *dust  = dust_GPU + blockIdx.x;
    g_dust_simpson_kernel(tuple, dust, division);
}

__device__ void
g_dust_simpson_kernel (float *tuple,
                       float *answer_GPU,
                       int division)
{
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    __shared__ float sdata1[TPB];
    __shared__ float sdata2[TPB];
    __shared__ float sdata3[TPB];

    float width = RANGE_WIDTH / (float)division;

    // MAP PHASE
    // put (f1, f2, f3) into (o1, o2, o3) for all samples
    for (int i = threadIdx.x; i < division; i += blockDim.x) {
        float window_left = width * i + RANGE_MIN;
        o1 += simpson_f1( window_left, width, tuple );
        o2 += simpson_f2( window_left, width, tuple );
        o3 += simpson_f3( window_left, width, tuple );
    }

    // REDUCE PHASE
    // Get sum of (o1, o2, o3) for all threads
    sdata1[threadIdx.x] = o1;
    sdata2[threadIdx.x] = o2;
    sdata3[threadIdx.x] = o3;
    g_reduceBlock<TPB>(sdata1, sdata2, sdata3);

    if (threadIdx.x == 0) {
        float int1 = sdata1[0];
        float int2 = sdata2[0];
        float int3 = sdata3[0];
        if (int1 < VERYSMALL) int1 = VERYSMALL;
        if (int2 < VERYSMALL) int2 = VERYSMALL;
        if (int3 < 0.0f)      int3 = 0.0f;

        float dust = -log10(int3 / (int1 * int2));

        if (dust < 0.0) { dust = 0.0f; }
        *answer_GPU = dust;
    }
}


template<unsigned int blockSize>
__device__ void
g_reduceBlock (float *sdata1, float *sdata2, float *sdata3)
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

//!
// With seq on global memory.
//
__global__ void
g_match (float *ts_GPU,
         float *tsc_GPU,
         float *dust_GPU,
         size_t ts_length,
         size_t ts_num,
         float *samples)
{
    int time = blockIdx.x;

    float sample1, sample2, sample3;
    int offset1 = blockIdx.x * INTEGRATION_SAMPLES;
    int offset2 = offset1 + INTEGRATION_SAMPLES * gridDim.x * ts_num;  // 50000 * lim * 1 + offset1
    int offset3 = offset2 + INTEGRATION_SAMPLES * gridDim.x * ts_num;  // 50000 * lim * 2 + offset1
    float *samples1 = &samples[offset1];
    float *samples2 = &samples[offset2];
    float *samples3 = &samples[offset3];

    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;

    __shared__ float sdata1[TPB];
    __shared__ float sdata2[TPB];
    __shared__ float sdata3[TPB];

    float *dusts = &dust_GPU[ts_num * time];
    float r = (float)RANGE_WIDTH / INTEGRATION_SAMPLES;

    float *tsc = &tsc_GPU[ts_num * time * 3];  // db for this block
    float *x  = &ts_GPU[time * 3];           // TODO: compute f1 only once.

    for (int i = 0; i < ts_num; i++) {
        float *y = &tsc[i * 3];
        o1 = o2 = o3 = 0.0f;
        for (int j = threadIdx.x; j < INTEGRATION_SAMPLES; j += blockDim.x) {
            sample1 = samples1[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;
            sample2 = samples2[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;
            sample3 = samples3[i * ts_num + j] * RANGE_WIDTH + RANGE_MIN;

            o1 += g_f12_multi( sample1, x );
            o2 += g_f12_multi( sample2, y );
            o3 += g_f3_multi( sample3, x, y );
        }

        sdata1[threadIdx.x] = o1;
        sdata2[threadIdx.x] = o2;
        sdata3[threadIdx.x] = o3;

        g_reduceBlock<TPB>(sdata1, sdata2, sdata3);

        __syncthreads();

        if (threadIdx.x == 0) {
            float int1 = sdata1[0] * r;
            float int2 = sdata2[0] * r;
            float int3 = sdata3[0] * r;
            if (int1 < VERYSMALL) int1 = VERYSMALL;
            if (int2 < VERYSMALL) int2 = VERYSMALL;
            if (int3 < 0.0f)      int3 = 0.0f;

            float dust = -log10(int3 / (int1 * int2));
            if (dust < 0.0) { dust = 0.0f; }

            dusts[i] = dust;
        }
    }
}


//!
// With seq on global memory.
//
__global__ void
g_match_simpson (float *ts_GPU,
                 float *tsc_GPU,
                 float *dust_GPU,
                 size_t ts_length,
                 size_t ts_num,
                 int division)
{
    int time = blockIdx.x;

    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    __shared__ float sdata1[TPB];
    __shared__ float sdata2[TPB];
    __shared__ float sdata3[TPB];

    float *dusts = &dust_GPU[ts_num * time];
    float *tsc = &tsc_GPU[ts_num * time * 3];  // TimeSeriesCollection for this block
    float *x  = &ts_GPU[time * 3];             // TODO: compute f1 only once.

    float width = RANGE_WIDTH / (float)division;

    for (int i = 0; i < ts_num; i++) {
        float *y = &tsc[i * 3];
        o1 = o2 = o3 = 0.0f;
        for (int j = threadIdx.x; j < division; j += blockDim.x) {
            float window_left = width * i + RANGE_MIN;
            o1 += simpson_f12_multi( window_left, width, x );
            o2 += simpson_f12_multi( window_left, width, y );
            o3 += simpson_f3_multi( window_left, width, x, y );
        }

        sdata1[threadIdx.x] = o1;
        sdata2[threadIdx.x] = o2;
        sdata3[threadIdx.x] = o3;
        g_reduceBlock<TPB>(sdata1, sdata2, sdata3);

        __syncthreads();

        if (threadIdx.x == 0) {
            float int1 = sdata1[0];
            float int2 = sdata2[0];
            float int3 = sdata3[0];
            if (int1 < VERYSMALL) int1 = VERYSMALL;
            if (int2 < VERYSMALL) int2 = VERYSMALL;
            if (int3 < 0.0f)      int3 = 0.0f;

            float dust = -log10(int3 / (int1 * int2));
            if (dust < 0.0) { dust = 0.0f; }

            dusts[i] = dust;
        }
    }
}


//
// for Test
//_____________

__global__ void
g_f123_test(float *xy, float *results)
{
    __syncthreads();

    float *x = xy;
    float *y = xy + 3;

    float v = (x[1] + y[1]) * 0.5;
    results[0] += g_f1(v, xy);
    results[1] += g_f12_multi(v, x);

    results[2] += g_f2(v, xy);
    results[3] += g_f12_multi(v, y);

    results[4] += g_f3(v, xy);
    results[5] += g_f3_multi(v, x, y);
}
