#include "kernel.hpp"
#include "randomvariable.hpp"
#include "config.hpp"

#include <stdio.h>

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


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
g_f1 (float v, float *params)
{
    float p1 = g_myPDF( params[ PARAM_X_DISTRIBUTION ],   // distribution
                        0.0f,                                // mean
                        params[ PARAM_X_STDDEV ],         // stddev
                        params[ PARAM_X_OBSERVATION ]-v ); // target

    float p2 = g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}


// calculate p(y|r(y)=v)p(r(y)=v)
__device__ float
g_f2 (float v, float *params)
{
    float p1 = g_myPDF( params[ PARAM_Y_DISTRIBUTION ],   // distribution
                        0.0f,                                // mean
                        params[ PARAM_Y_STDDEV ],         // stddev
                        params[ PARAM_Y_OBSERVATION ] - v );  // target

    float p2 = g_pdf_uniform( -RANGE_VALUE, RANGE_VALUE, v );

    return p1 * p2;
}


// p(r(x)=z|x) * p(r(y)=z|y)
__device__ float
g_f3 (float z, float *params)
{
    int   x_dist   = (int)params[ PARAM_X_DISTRIBUTION ];
    float x        =      params[ PARAM_X_OBSERVATION ] - 0.1f;
    float x_stddev =      params[ PARAM_X_STDDEV ];
    int   y_dist   = (int)params[ PARAM_Y_DISTRIBUTION ];
    float y        =      params[ PARAM_Y_OBSERVATION ] + 0.1f;
    float y_stddev =      params[ PARAM_Y_STDDEV ];

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
g_f4 (float k, float *params)
{
    return 1.0f;
}


// With seq on global memory
__global__ void
g_distance_kernel (float *seq_GPU,
                   float *samples_GPU,
                   float *dust_GPU)
{
    float *p_param = seq_GPU  + blockIdx.x * PARAM_SIZE;
    float *p_dust  = dust_GPU + blockIdx.x;

    g_dust_kernel(p_param, samples_GPU, p_dust);
}


__device__ void
g_dust_kernel (float *params,
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
        o1 += g_f1( in1, params );
        o2 += g_f2( in2, params );
        o3 += g_f3( in3, params );
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


// With seq on global memory
__global__ void
g_match (float *ts_GPU,
         float *db_GPU,
         float *dust_GPU,
         size_t ts_length,
         size_t db_num,
         float *in)
{
    int time = blockIdx.x;

    float in1, in2, in3;
    int offset1 = blockIdx.x * INTEGRATION_SAMPLES;
    int offset2 = offset1 + INTEGRATION_SAMPLES * gridDim.x;  // 50000 * lim * 1 + offset1
    int offset3 = offset2 + INTEGRATION_SAMPLES * gridDim.x;  // 50000 * lim * 2 + offset1
    float *samples1 = &in[offset1];
    float *samples2 = &in[offset2];
    float *samples3 = &in[offset3];

    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;

    __shared__ float sdata1[TPB];
    __shared__ float sdata2[TPB];
    __shared__ float sdata3[TPB];

    float *dusts = &dust_GPU[db_num * time];
    float r = (float)RANGE_WIDTH / INTEGRATION_SAMPLES;

    float *db = &db_GPU[db_num * time * 3];  // db for this block
    float *x  = &ts_GPU[time * 3];           // TODO: compute f1 only once.

    for (int i = 0; i < db_num; i++) {
        float *y = &db[i * 3];
        o1 = o2 = o3 = 0.0f;
        for (int j = threadIdx.x; j < INTEGRATION_SAMPLES; j += blockDim.x) {
            in1 = samples1[j] * RANGE_WIDTH + RANGE_MIN;
            in2 = samples2[j] * RANGE_WIDTH + RANGE_MIN;
            in3 = samples3[j] * RANGE_WIDTH + RANGE_MIN;

            o1 += g_f12_multi( in1, x );
            o2 += g_f12_multi( in2, y );
            o3 += g_f3_multi( in3, x, y );
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

            // printf("%d:%d: \t %f \t %f \t %f\n", i, time, int1, int2, int3);

            dusts[i] = dust;
        }
    }
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

__global__ void
g_f123_test(float *param, float *results)
{
    __syncthreads();

    float *x = param;
    float *y = param + 3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float v = (x[1] + y[1]) * 0.5;
    results[0] += g_f1(v, param);
    results[1] += g_f12_multi(v, x);

    results[2] += g_f2(v, param);
    results[3] += g_f12_multi(v, y);

    results[4] += g_f3(v, param);
    results[5] += g_f3_multi(v, x, y);
}
