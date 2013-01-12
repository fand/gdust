#pragma once

//#include <cuda.h>
//#include <curand.h>

//#include <cassert>

//#include <thrust/random/linear_congruential_engine.h>
//#include <thrust/random/uniform_real_distribution.h>
//#include <thrust/random/normal_distribution.h>


__device__ bool check_uniform_lower(float lower, float *result);
__device__ bool check_uniform_upper(float upper, float *result);
__device__ bool check_uniform(float lower, float upper, float *result);
__device__ bool check_uniform_x(float const& x, float *result);    
__device__ bool check_location(float location, float * result);
__device__ bool check_x(float x, float *result);
__device__ bool check_scale(float scale, float *result);
__device__ bool verify_lambda(float l, float *result);
__device__ bool verify_exp_x(float x, float *result);

//__device__ float pdf( thrust::uniform_real_distribution<float> dist, float x );
//__device__ float pdf( thrust::random::experimental::normal_distribution<float> dist, float x );

__device__ float pdf_uniform( float lower, float upper, float x );
__device__ float pdf_normal( float mean, float sd, float x );

__device__ float myPDF( int distribution, float mean, float stddev, float v );

__device__ float f1 ( float *k, size_t dim, void *params );
__device__ float f2 ( float *k, size_t dim, void *params );
__device__ float f3 ( float *k, size_t dim, void *params );
__device__ float f4 ( float *k, size_t dim, void *params );

__global__ void integrate_kernel( void *pair, int fnum, float *input_array, float *output_array );
