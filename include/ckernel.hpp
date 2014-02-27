#pragma once

#define PARAM_SIZE 6
#define PARAM_X_DISTRIBUTION 0
#define PARAM_X_OBSERVATION  1
#define PARAM_X_STDDEV       2
#define PARAM_Y_DISTRIBUTION 3
#define PARAM_Y_OBSERVATION  4
#define PARAM_Y_STDDEV       5

#define PI_DOUBLE 3.14159265358979323846264338327950288

double c_pdf_uniform (double lower, double upper, double x);
double c_pdf_normal (double mean, double sd, double x);
double c_myPDF (int distribution, double mean, double stddev, double v);
double c_f1 (double v, double *params);
double c_f2 (double v, double *params);
double c_f3 (double z, double *params);
double c_f4 (double k, double *params);
double c_dust_kernel (double *params, int time);

double myrand();
