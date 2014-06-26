#pragma once

#define PARAM_SIZE 6
#define PARAM_X_DISTRIBUTION 0
#define PARAM_X_OBSERVATION  1
#define PARAM_X_STDDEV       2
#define PARAM_Y_DISTRIBUTION 3
#define PARAM_Y_OBSERVATION  4
#define PARAM_Y_STDDEV       5



__device__ float g_pdf_uniform (float lower, float upper, float x);
__device__ float g_pdf_normal (float mean, float sd, float x);

__device__ float g_myPDF (int distribution, float mean, float stddev, float v );

__device__ float g_f1 (float v, float *params);
__device__ float g_f2 (float v, float *params);
__device__ float g_f3 (float z, float *params);
__device__ float g_f4 (float k, float *params);

__global__ void g_distance_kernel (float *seq_GPU, float *samples_GPU, float *dust_GPU);
__device__ void g_dust_kernel (float *params, float *in, float *answer_GPU);

template<unsigned int blockSize> __device__ void g_reduceBlock (float *sdata1,
                                                                float *sdata2,
                                                                float *sdata3);

__global__ void g_match (float *ts_GPU, float *db_GPU, float *DUST_GPU, size_t ts_length, size_t db_num,
                         float *o1, float *o2, float *o3, float *samples_GPU);

__device__ float g_f12_multi (float v, float *x);
__device__ float g_f3_multi (float z, float *x, float *y);

__global__ void g_f123_test(float *param, float *results);
