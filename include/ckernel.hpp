#pragma once

#define TUPLE_SIZE 6
#define TUPLE_X_DISTRIBUTION 0
#define TUPLE_X_OBSERVATION  1
#define TUPLE_X_STDDEV       2
#define TUPLE_Y_DISTRIBUTION 3
#define TUPLE_Y_OBSERVATION  4
#define TUPLE_Y_STDDEV       5

#define PI_DOUBLE 3.14159265358979323846264338327950288

float c_pdf_uniform(float lower, float upper, float x);
float c_pdf_normal(float mean, float sd, float x);
float c_myPDF(int distribution, float mean, float stddev, float v);
float c_f1(float v, float *xy);
float c_f2(float v, float *xy);
float c_f3(float z, float *xy);
float c_f4(float k, float *xy);
float c_dust_kernel(float *xy, float *samples, int time);
