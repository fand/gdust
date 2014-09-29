#pragma once

#include "dust_inner.hpp"


#define TUPLE_SIZE 6
#define TUPLE_X_DISTRIBUTION 0
#define TUPLE_X_OBSERVATION  1
#define TUPLE_X_STDDEV       2
#define TUPLE_Y_DISTRIBUTION 3
#define TUPLE_Y_OBSERVATION  4
#define TUPLE_Y_STDDEV       5


__global__ void g_f123_test(float *param, float *results);

__device__ float simpson_f1(float left, float width, float *tuple);
__device__ float simpson_f2(float left, float width, float *tuple);
__device__ float simpson_f3(float left, float width, float *tuple);
__device__ float simpson_f12_multi(float left, float width, float *x);
__device__ float simpson_f3_multi(float left, float width, float *x, float *y);

__device__ inline float
simpson_f1(float left, float width, float *tuple) {
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (f1(left, tuple) +
                        f1(right, tuple) +
                        f1(mid, tuple) * 4);
}
__device__ inline float
simpson_f2(float left, float width, float *tuple) {
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (f2(left, tuple) +
                        f2(right, tuple) +
                        f2(mid, tuple) * 4);
}
__device__ inline float
simpson_f3(float left, float width, float *tuple) {
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (f3(left, tuple) +
                        f3(right, tuple) +
                        f3(mid, tuple) * 4);
}
__device__ inline float
simpson_f12_multi(float left, float width, float *x) {
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (f12_multi(left, x) +
                        f12_multi(right, x) +
                        f12_multi(mid, x) * 4);
}
__device__ inline float
simpson_f3_multi(float left, float width, float *x, float *y) {
  float mid = left + width * 0.5;
  float right = left + width;
  return (width / 6) * (f3_multi(left, x, y) +
                        f3_multi(right, x, y) +
                        f3_multi(mid, x, y) * 4);
}
