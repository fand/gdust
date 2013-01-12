#pragma once

#include <curand.h>

#define INTEGRATION_SAMPLES 50000


class Integrator
{
public:
    ~Integrator();
    Integrator();    

    float integrate( int fnum, float *param );

private:
    float *in, *out, *param;
    float *in_GPU, *out_GPU, *param_GPU;
    curandGenerator_t *gen;
};

