#include "config.hpp"
#include "RandomVariable.hpp"

__host__ __device__ float pdf_uniform(float lower, float upper, float x);
__host__ __device__ float pdf_normal(float mean, float sd, float x);
__host__ __device__ float myPDF(int distribution, float mean, float stddev, float v);

__host__ __device__ float f1(float v, float *xy);
__host__ __device__ float f2(float v, float *xy);
__host__ __device__ float f3(float z, float *xy);
__host__ __device__ float f12_multi(float v, float *x);
__host__ __device__ float f3_multi(float z, float *x, float *y);


//
// PDF functions
// __________________

__host__ __device__ inline float
pdf_uniform(float lower, float upper, float x) {
  if ((x < lower) || (x > upper)) {
    return 0.0f;
  }
  if (lower == x && upper == x) {
    return 0.0f;
  }
  return 1.0f / (upper - lower);
}

__host__ __device__ inline float
pdf_normal(float mean, float sd, float x) {
  if (isinf(x) || sd <= 0 || isinf(sd) || isinf(mean)) {
    return 0.0f;
  }

  float result = 0.0f;

  float exponent = x - mean;
  exponent *= -exponent;
  exponent /= (2 * sd * sd);

#if defined(__CUDA_ARCH__)
  // Device code here
  result = __expf(exponent);
#else
  // Host code here
  result = exp(exponent);
#endif
  result /= sd * sqrt(2 * PI_FLOAT);

  return result;
}

__host__ __device__ inline float
myPDF(int distribution, float mean, float stddev, float v) {
  float ret = -1.0f;
  if (stddev == 0.0f) stddev = 0.2f;

  if (distribution == RANDVAR_UNIFORM) {
    float b = SQRT3 * stddev;
    ret = pdf_uniform(-b, b, v);
  } else if (distribution == RANDVAR_NORMAL) {
    ret = pdf_normal(0, 1, v / stddev);
  }

  return ret;
}


//
// Integrand functions in dust
// ________________________________

//!
// Calculate p(x|r(x)=v)p(r(x)=v).
//
// @param {float}   v  - Random value
// @param {float[]} xy - An array containing x & y
__host__ __device__ inline float
f1(float v, float *xy) {
  float p1 = myPDF(xy[ TUPLE_X_DISTRIBUTION ],    // distribution
                     0.0f,                          // mean
                     xy[ TUPLE_X_STDDEV ],          // stddev
                     xy[ TUPLE_X_OBSERVATION ]-v);  // target

  float p2 = pdf_uniform(-RANGE_VALUE, RANGE_VALUE, v);

  return p1 * p2;
}

//!
// Calculate p(y|r(y)=v)p(r(y)=v).
// Almost same as f1.
//
__host__ __device__ inline float
f2(float v, float *xy) {
  float p1 = myPDF(xy[ TUPLE_Y_DISTRIBUTION ],      // distribution
                     0.0f,                            // mean
                     xy[ TUPLE_Y_STDDEV ],            // stddev
                     xy[ TUPLE_Y_OBSERVATION ] - v);  // target

  float p2 = pdf_uniform(-RANGE_VALUE, RANGE_VALUE, v);

  return p1 * p2;
}

//!
// Calculate p(r(x)=z|x) * p(r(y)=z|y).
//
// @param {float}   z  - Random value
// @param {float[]} xy - An array containing x & y
//
__host__ __device__ inline float
f3(float z, float *xy) {
  int   x_dist   = static_cast<int>(xy[TUPLE_X_DISTRIBUTION]);
  float x        = xy[TUPLE_X_OBSERVATION] - 0.1f;
  float x_stddev = xy[TUPLE_X_STDDEV];
  int   y_dist   = static_cast<int>(xy[TUPLE_Y_DISTRIBUTION]);
  float y        = xy[TUPLE_Y_OBSERVATION] + 0.1f;
  float y_stddev = xy[TUPLE_Y_STDDEV];

  float p1, p2;

  if (x_dist == RANDVAR_UNIFORM) {
    float x_adjust = 0;
    float y_adjust = 0;

    if (abs(x-z) > x_stddev * SQRT3) {
      x_adjust = myPDF(x_dist, 0, x_stddev, 0) *
        (1 + erf(-(abs(x-z) - x_stddev * SQRT3)));
    }

    if (abs(y-z) > y_stddev * SQRT3) {
      y_adjust = myPDF(y_dist, 0, y_stddev, 0) *
        (1 + erf(-(abs(y-z) - y_stddev * SQRT3)));
    }

    float pdf_x = myPDF(x_dist, 0.0f, x_stddev, x-z) + x_adjust;
    float pdf_y = myPDF(y_dist, 0.0f, y_stddev, y-z) + y_adjust;

    p1 = pdf_x * pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
    p2 = pdf_y * pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
  } else {
    // p(r(x)=z|x) and p(r(y)=z|y)
    p1 = (myPDF(x_dist, 0, x_stddev, x-z) *
          pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
    p2 = (myPDF(y_dist, 0, y_stddev, y-z) *
          pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
  }

  return p1 * p2;
}

// calculate p(y|r(y)=v)p(r(y)=v)
__host__ __device__ inline float
f12_multi(float v, float *x) {
  float p1 = myPDF(x[0], 0.0f, x[2], x[1] - v);
  float p2 = pdf_uniform(-RANGE_VALUE, RANGE_VALUE, v);
  return p1 * p2;
}

// p(r(x)=z|x) * p(r(y)=z|y)
__host__ __device__ inline float
f3_multi(float z, float *x_, float *y_) {
  int   x_dist   = static_cast<int>(x_[0]);
  float x        = x_[1] - 0.1f;
  float x_stddev = x_[2];
  int   y_dist   = static_cast<int>(y_[0]);
  float y        = y_[1] + 0.1f;
  float y_stddev = y_[2];

  float p1, p2;

  if (x_dist == RANDVAR_UNIFORM) {
    float x_adjust = 0;
    float y_adjust = 0;

    if (abs(x-z) > x_stddev * SQRT3) {
      x_adjust = myPDF(x_dist, 0, x_stddev, 0) *
        (1 + erf(-(abs(x-z) - x_stddev * SQRT3)));
    }

    if (abs(y-z) > y_stddev * SQRT3) {
      y_adjust = myPDF(y_dist, 0, y_stddev, 0) *
        (1 + erf(-(abs(y-z) - y_stddev * SQRT3)));
    }

    float pdf_x = myPDF(x_dist, 0.0f, x_stddev, x-z) + x_adjust;
    float pdf_y = myPDF(y_dist, 0.0f, y_stddev, y-z) + y_adjust;

    p1 = pdf_x * pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
    p2 = pdf_y * pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
  } else {
    // p(r(x)=z|x) and p(r(y)=z|y)
    p1 = (myPDF(x_dist, 0, x_stddev, x-z) *
          pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
    p2 = (myPDF(y_dist, 0, y_stddev, y-z) *
          pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
  }

  return p1 * p2;
}
