#include <stdio.h>
#include <cutil.h>
#include "randomvariable.hpp"

#include <assert.h>

#define NEAR(x,y) (assert( abs((x) - (y)) < 0.1 ))


#include "kernel.cu"
#include "dust.hpp"


__global__ void ggf1(float v, float *params, float *result){ *result = g_f1(v, params); }
__global__ void ggf2(float v, float *params, float *result){ *result = g_f2(v, params); }
__global__ void ggf3(float v, float *params, float *result){ *result = g_f3(v, params); }


void TestF1(float v, float *params);
void TestF2(float v, float *params);
void TestF3(float v, float *params);

int main()
{

    for (float i = 0; i < 10; i++) {
        float params[] = {1, 0.1, 0.1,
                          2, 0.1, 0.1};
        float v = i / 10.0f;
        
        TestF1(v, params);
        TestF2(v, params);
        TestF3(v, params);
    }

    printf("Test passed! :D\n");
}


void
TestF1(float v, float *params)
{
    float cpu, gpu;

    // CPU
    RandomVariable x(params[0], params[1], params[1], params[2]);   
    RandomVariable y(params[3], params[4], params[4], params[5]);
    RandomVariable *pair[2] = {&x, &y};
    double vv = (double)v;
    cpu = (float)f1(&vv, 1, (void*)pair);

    // GPU
    float *params_D, *result_D;
    cudaMalloc((void**)&params_D, sizeof(float) * 6);
    cudaMalloc((void**)&result_D, sizeof(float) * 1);    

    cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice);
    ggf1<<< 1, 1 >>>(v, params_D, result_D);
    cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost);

    NEAR(cpu, gpu);
}


void
TestF2(float v, float *params)
{
    float cpu, gpu;

    // CPU
    RandomVariable x(params[0], params[1], params[1], params[2]);   
    RandomVariable y(params[3], params[4], params[4], params[5]);
    RandomVariable *pair[2] = {&x, &y};
    double vv = (double)v;
    cpu = (float)f2(&vv, 1, (void*)pair);

    // GPU
    float *params_D, *result_D;
    cudaMalloc((void**)&params_D, sizeof(float) * 6);
    cudaMalloc((void**)&result_D, sizeof(float) * 1);    

    cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice);
    ggf2<<< 1, 1 >>>(v, params_D, result_D);
    cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost);

    NEAR(cpu, gpu);
}


void
TestF3(float v, float *params)
{
    float cpu, gpu;

    // CPU
    double cparams[] = {1.0, 1.0,
                        params[0], params[1], params[2],
                        params[3], params[4], params[5]};
    double vv = (double)v;
    cpu = (float)f3(&vv, 1, (void*)cparams);

    // GPU
    float *params_D, *result_D;
    cudaMalloc((void**)&params_D, sizeof(float) * 6);
    cudaMalloc((void**)&result_D, sizeof(float) * 1);    

    cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice);
    ggf3<<< 1, 1 >>>(v, params_D, result_D);
    cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost);

    NEAR(cpu, gpu);
}

