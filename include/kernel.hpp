#pragma once

#define TPB 512
#define INTEGRATION_SAMPLES 49152
#define RANGE_MIN -16
#define RANGE_MAX 16
#define RANGE_WIDTH 32


__device__ bool check_uniform_lower(float lower, float *result);
__device__ bool check_uniform_upper(float upper, float *result);
__device__ bool check_uniform(float lower, float upper, float *result);
__device__ bool check_uniform_x(float const& x, float *result);    
__device__ bool check_location(float location, float * result);
__device__ bool check_x(float x, float *result);
__device__ bool check_scale(float scale, float *result);
__device__ bool verify_lambda(float l, float *result);
__device__ bool verify_exp_x(float x, float *result);

__device__ float g_pdf_uniform( float lower, float upper, float x );
__device__ float g_pdf_normal( float mean, float sd, float x );

__device__ float g_myPDF( int distribution, float mean, float stddev, float v );

__device__ float g_f1 ( float v, float *params );
__device__ float g_f2 ( float v, float *params );
__device__ float g_f3 ( float z, float *params );
__device__ float g_f4 ( float k, float *params );


__global__ void g_distance_kernel(float *seq_GPU, float *samples_GPU, float *dust_GPU);
__device__ void g_dust_kernel(float *params, float *in, float *answer_GPU);

template<unsigned int blockSize> __device__ void g_reduceBlock(float *sdata1,
                                                               float *sdata2,
                                                               float *sdata3);
