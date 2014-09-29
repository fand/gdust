#include "ckernel.hpp"
#include "dust_inner.hpp"
#include "RandomVariable.hpp"
#include "config.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <omp.h>


//
// PDF functions
// __________________

inline float
c_pdf_uniform(float lower, float upper, float x) {
  if ((x < lower) || (x > upper)) {
    return 0.0;
  }
  if (lower == x && upper == x) {
    return 0.0;
  }
  return 1.0 / (upper - lower);
}

inline float
c_pdf_normal(float mean, float sd, float x) {
  if (isinf(x) || sd <= 0 || isinf(sd) || isinf(mean)) {
    return 0.0;
  }

  float result = 0.0;

  float exponent = x - mean;
  exponent *= -exponent;
  exponent /= (2 * sd * sd);

  result = exp(exponent);
  result /= sd * sqrt(2 * PI_FLOAT);

  return result;
}

inline float
c_myPDF(int distribution, float mean, float stddev, float v) {
  float ret = -1.0f;
  if (stddev == 0.0f) stddev = 0.2f;

  if (distribution == RANDVAR_UNIFORM) {
    float b = SQRT3 * stddev;
    ret = c_pdf_uniform(-b, b, v);
  } else if (distribution == RANDVAR_NORMAL) {
    ret = c_pdf_normal(0, 1, v / stddev);
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
inline float
c_f1(float v, float *xy) {
  float p1 = c_myPDF(xy[ TUPLE_X_DISTRIBUTION ],     // distribution
                      0.0,                            // mean
                      xy[ TUPLE_X_STDDEV ],           // stddev
                      xy[ TUPLE_X_OBSERVATION ]-v);   // target

  float p2 = c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, v);

  return p1 * p2;
}

//!
// Calculate p(y|r(y)=v)p(r(y)=v).
// Almost same as c_f1.
//
inline float
c_f2(float v, float *xy) {
  float p1 = c_myPDF(xy[ TUPLE_Y_DISTRIBUTION ],       // distribution
                      0.0,                              // mean
                      xy[ TUPLE_Y_STDDEV ],             // stddev
                      xy[ TUPLE_Y_OBSERVATION ] - v);   // target

  float p2 = c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, v);

  return p1 * p2;
}

//!
// Calculate p(r(x)=z|x) * p(r(y)=z|y).
//
// @param {float}   z  - Random value
// @param {float[]} xy - An array containing x & y
//
inline float
c_f3(float z, float *xy) {
  int    x_dist   = static_cast<int>(xy[ TUPLE_X_DISTRIBUTION ]);
  float x        = xy[ TUPLE_X_OBSERVATION ] - 0.1;
  float x_stddev = xy[ TUPLE_X_STDDEV ];
  int    y_dist   = static_cast<int>(xy[ TUPLE_Y_DISTRIBUTION ]);
  float y        = xy[ TUPLE_Y_OBSERVATION ] + 0.1;
  float y_stddev = xy[ TUPLE_Y_STDDEV ];

  float p1, p2;

  if (x_dist == RANDVAR_UNIFORM) {
    float x_adjust = 0;
    float y_adjust = 0;

    if (abs(x-z) > x_stddev * SQRT3) {
      x_adjust = c_myPDF(x_dist, 0, x_stddev, 0) *
        (1 + erf(-(abs(x-z) - x_stddev * SQRT3)));
    }

    if (abs(y-z) > y_stddev * SQRT3) {
      y_adjust = c_myPDF(y_dist, 0, y_stddev, 0) *
        (1 + erf(-(abs(y-z) - y_stddev * SQRT3)));
    }

    float pdf_x = c_myPDF(x_dist, 0.0, x_stddev, x-z) + x_adjust;
    float pdf_y = c_myPDF(y_dist, 0.0, y_stddev, y-z) + y_adjust;

    p1 = pdf_x * c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
    p2 = pdf_y * c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z);
  } else {
    // p(r(x)=z|x) and p(r(y)=z|y)
    p1 = (c_myPDF(x_dist, 0, x_stddev, x-z) *
           c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
    p2 = (c_myPDF(y_dist, 0, y_stddev, y-z) *
           c_pdf_uniform(-RANGE_VALUE, RANGE_VALUE, z));
  }

  return p1 * p2;
}

float
c_f4(float k, float *xy) {
  return 1.0;
}


//
// Functions for dust / DUST
// ______________________________

float
c_dust_kernel(float *xy, float *samples, int time) {
  float o1 = 0.0;
  float o2 = 0.0;
  float o3 = 0.0;

  int offset = time * 3 * INTEGRATION_SAMPLES;
  float *local_samples = samples + offset;
  for (int i = 0; i < INTEGRATION_SAMPLES; ++i) {
    o1 += (float)f1(local_samples[i * 3    ], xy);
    o2 += (float)f2(local_samples[i * 3 + 1], xy);
    o3 += (float)f3(local_samples[i * 3 + 2], xy);
  }

  float r = static_cast<float>(RANGE_WIDTH) / INTEGRATION_SAMPLES;
  float int1 = o1 * r;
  float int2 = o2 * r;
  float int3 = o3 * r;
  if (int1 < VERYSMALL) int1 = VERYSMALL;
  if (int2 < VERYSMALL) int2 = VERYSMALL;
  if (int3 < 0.0) int3 = 0.0;

  float d = -log10(int3 / (int1 * int2));

  if (d < 0.0) { d = 0.0; }

  return d;
}
