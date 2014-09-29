#pragma once

#define TUPLE_SIZE 6
#define TUPLE_X_DISTRIBUTION 0
#define TUPLE_X_OBSERVATION  1
#define TUPLE_X_STDDEV       2
#define TUPLE_Y_DISTRIBUTION 3
#define TUPLE_Y_OBSERVATION  4
#define TUPLE_Y_STDDEV       5

#define PI_DOUBLE 3.14159265358979323846264338327950288

double c_pdf_uniform(double lower, double upper, double x);
double c_pdf_normal(double mean, double sd, double x);
double c_myPDF(int distribution, double mean, double stddev, double v);
double c_f1(double v, double *xy);
double c_f2(double v, double *xy);
double c_f3(double z, double *xy);
double c_f4(double k, double *xy);
double c_dust_kernel(double *xy, double *samples, int time);
