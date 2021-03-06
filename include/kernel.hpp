#pragma once

#define TUPLE_SIZE 6
#define TUPLE_X_DISTRIBUTION 0
#define TUPLE_X_OBSERVATION  1
#define TUPLE_X_STDDEV       2
#define TUPLE_Y_DISTRIBUTION 3
#define TUPLE_Y_OBSERVATION  4
#define TUPLE_Y_STDDEV       5


__device__ float g_pdf_uniform(float lower, float upper, float x);
__device__ float g_pdf_normal(float mean, float sd, float x);
__device__ float g_myPDF(int distribution, float mean, float stddev, float v);

__device__ float g_f1(float v, float *xy);
__device__ float g_f2(float v, float *xy);
__device__ float g_f3(float z, float *xy);
__device__ float g_f4(float k, float *xy);
__device__ float g_f12_multi(float v, float *x);
__device__ float g_f3_multi(float z, float *x, float *y);
__device__ float simpson_f1(float left, float width, float *tuple);
__device__ float simpson_f2(float left, float width, float *tuple);
__device__ float simpson_f3(float left, float width, float *tuple);
__device__ float simpson_f12_multi(float left, float width, float *x);
__device__ float simpson_f3_multi(float left, float width, float *x, float *y);

__global__ void g_distance_kernel(float *tsc_GPU, float *samples_GPU, float *dust_GPU);
__device__ void g_dust_kernel(float *tuple, float *samples, float *answer_GPU);
__global__ void g_distance_simpson_kernel(float *tuples__GPU, float *dust_GPU, int division);
__device__ void g_dust_simpson_kernel(float *tuple, float *dust_GPU, int division);

template <unsigned int blockSize> __device__ void g_reduceBlock(float *sdata1,
                                                               float *sdata2,
                                                               float *sdata3);

__global__ void g_match(float *ts_GPU, float *tsc_GPU, float *DUST_GPU, size_t ts_length, size_t ts_num,
                        float *samples_GPU);
__global__ void g_match_simpson(float *ts_GPU, float *tsc_GPU, float *DUST_GPU, size_t ts_length, size_t ts_num,
                                int division);

__global__ void g_f123_test(float *param, float *results);
