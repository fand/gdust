#include "kernel.hpp"

//
// for Test
// _____________

__global__ void
g_f123_test(float *xy, float *results) {
  __syncthreads();

  float *x = xy;
  float *y = xy + 3;

  float v = (x[1] + y[1]) * 0.5;
  results[0] += f1(v, xy);
  results[1] += f12_multi(v, x);

  results[2] += f2(v, xy);
  results[3] += f12_multi(v, y);

  results[4] += f3(v, xy);
  results[5] += f3_multi(v, x, y);
}
