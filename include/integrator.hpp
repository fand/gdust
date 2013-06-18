#pragma once

#include <curand.h>


class Integrator
{
public:
    ~Integrator();
    Integrator();    

    float integrate( int fnum, float *param );
    float phi( float *param );
    
private:
    float *in, *out, *sum;
    float *in_GPU, *out_GPU, *param_GPU, *sum_GPU;
    curandGenerator_t *gen;

    float *sum1, *sum2, *sum3;
    float *out1_GPU, *out2_GPU, *out3_GPU;
    float *sum1_GPU, *sum2_GPU, *sum3_GPU;
    float *answer_GPU;
};

