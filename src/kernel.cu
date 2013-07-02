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


__device__ float pdf_uniform(float lower, float upper, float x)
{
    return ((lower <= x) && (x <= upper)) ? 1.0f / (upper - lower) : 0.0f;
}


__device__ float pdf_normal( float mean, float sd, float x )
{
    if (isinf(x)) return 0; // pdf(infinity) is zero.

    float result = 0.0f;

    float exponent = x - mean;
    exponent *= ((-1.0f) * exponent);
    exponent /= (2.0f * sd * sd);

    result = __expf(exponent);
    result /= sd * sqrt(2.0f * PI_FLOAT);

    return result;
}


template<int distribution>
__device__ float
myPDF(float mean, float stddev, float v)
{
    if (stddev == 0.0f) {
        stddev = 0.2f;
    }

    if (distribution == RANDVAR_UNIFORM) {
        float b = SQRT3 * stddev;
        return pdf_uniform(-b, b, v);
    } else if (distribution == RANDVAR_NORMAL) {
        return pdf_normal(0.0f, 1.0f, v / stddev);
    } else {
        return -1.0f;
    }
}


// calculate p(y|r(y)=v)p(r(y)=v)
template<int dist>
__device__ float
f12(float v, float ob, float std)
{
    return myPDF<dist>(0.0f, std, ob - v);
}

// p(r(x)=z|x) * p(r(y)=z|y)
template<int x_dist, int y_dist>
__device__ float
f3(float z, float x, float x_stddev, float y, float y_stddev)
{
    float px = myPDF<x_dist>(0.0f, x_stddev, x - z);
    float py = myPDF<y_dist>(0.0f, y_stddev, y - z);
    if (x_dist == RANDVAR_UNIFORM) {
        if (fabs(x - z) > x_stddev * SQRT3) {
            px += myPDF<x_dist>(0.0f, x_stddev, 0.0f) *
                (1.0f + erff(x_stddev * SQRT3 - fabs(x - z)));
        }

        if (fabs(y - z) > y_stddev * SQRT3) {
            py += myPDF<y_dist>(0.0f, y_stddev, 0.0f) *
                (1.0f + erff(y_stddev * SQRT3 - fabs(y - z)));
        }
    }
    return px * py;
}


template<int x_dist, int y_dist> __device__ float3
map(float x, float x_stddev, float y, float y_stddev, float *in)
{
    float3 o = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = threadIdx.x; i < INTEGRATION_SAMPLES; i += blockDim.x) {
        float input = __fmaf_rn(in[i], RANGE_WIDTH, RANGE_MIN);
        float u = pdf_uniform(-RANGE_VALUE, RANGE_VALUE, input);
        o.x = __fmaf_rn(u, f12<x_dist>(input, x, x_stddev), o.x);
        o.y = __fmaf_rn(u, f12<y_dist>(input, y, y_stddev), o.y);
        o.z = __fmaf_rn(u * u, f3<x_dist, y_dist>(input, x - 0.1f, x_stddev, y + 0.1f, y_stddev), o.z);
    }
    return o;
}


__global__ void distance_kernel(
    float *in_GPU,
    float *dust_GPU,
    float4 *seq_GPU,
    int *xy_dists
)
{
    float3 mySum;

    // MAP PHASE
    {
        int xy_dist = xy_dists[blockIdx.x];
        float4 params = seq_GPU[blockIdx.x];
        switch(xy_dist) {
        case 0x5:
            mySum = map<RANDVAR_UNIFORM, RANDVAR_UNIFORM>(params.x, params.y, params.z, params.w, in_GPU);
            break;
        case 0x6:
            mySum = map<RANDVAR_UNIFORM, RANDVAR_NORMAL>(params.x, params.y, params.z, params.w, in_GPU);
            break;
        case 0x9:
            mySum = map<RANDVAR_NORMAL, RANDVAR_UNIFORM>(params.x, params.y, params.z, params.w, in_GPU);
            break;
        case 0xa:
            mySum = map<RANDVAR_NORMAL, RANDVAR_NORMAL>(params.x, params.y, params.z, params.w, in_GPU);
            break;
        }
    }

    // REDUCE PHASE
    __shared__ float sdata[3][TPB];
    sdata[0][threadIdx.x] = mySum.x;
    sdata[1][threadIdx.x] = mySum.y;
    sdata[2][threadIdx.x] = mySum.z;
    __syncthreads();
    reduceBlock<TPB>(sdata[0], sdata[1], sdata[2], mySum.x, mySum.y, mySum.z);

    if (threadIdx.x == 0) {
        float d = sdata[2][0] / (sdata[0][0] * sdata[1][0] * RANGE_WIDTH / INTEGRATION_SAMPLES);
        dust_GPU[blockIdx.x] = d < 1.0f ? -__log10f(d) : 0.0f;
    }
}


template<unsigned int blockSize>
__device__ void
reduceBlock(float *sdata1, float *sdata2, float *sdata3, float mySum1, float mySum2, float mySum3)
{
    // do reduction in shared mem
    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            sdata1[threadIdx.x] = mySum1 = mySum1 + sdata1[threadIdx.x + 256];
            sdata2[threadIdx.x] = mySum2 = mySum2 + sdata2[threadIdx.x + 256];
            sdata3[threadIdx.x] = mySum3 = mySum3 + sdata3[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            sdata1[threadIdx.x] = mySum1 = mySum1 + sdata1[threadIdx.x + 128];
            sdata2[threadIdx.x] = mySum2 = mySum2 + sdata2[threadIdx.x + 128];
            sdata3[threadIdx.x] = mySum3 = mySum3 + sdata3[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (threadIdx.x <  64) {
            sdata1[threadIdx.x] = mySum1 = mySum1 + sdata1[threadIdx.x + 64];
            sdata2[threadIdx.x] = mySum2 = mySum2 + sdata2[threadIdx.x + 64];
            sdata3[threadIdx.x] = mySum3 = mySum3 + sdata3[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *smem1 = sdata1;
        volatile float *smem2 = sdata2;
        volatile float *smem3 = sdata3;

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 32];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 32];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 32];

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 16];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 16];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 16];

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 8];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 8];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 8];

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 4];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 4];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 4];

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 2];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 2];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 2];

        smem1[threadIdx.x] = mySum1 = mySum1 + smem1[threadIdx.x + 1];
        smem2[threadIdx.x] = mySum2 = mySum2 + smem2[threadIdx.x + 1];
        smem3[threadIdx.x] = mySum3 = mySum3 + smem3[threadIdx.x + 1];
    }
}
