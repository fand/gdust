#pragma once

#include <curand.h>


class Integrator
{
public:
    ~Integrator();
    Integrator();    

    float integrate( int fnum, float *param );

private:
    float *in, *out, *param, *sum;
    float *in_GPU, *out_GPU, *param_GPU, *sum_GPU;
    curandGenerator_t *gen;
};

