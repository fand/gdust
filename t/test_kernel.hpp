#include "kernel.cu"
#include "randomvariable.hpp"
#include "dust.hpp"

#include <boost/math/distributions/uniform.hpp> // for uniform distribution
#include <boost/math/distributions/normal.hpp> // for normal distribution
#include <boost/math/distributions/exponential.hpp> // for normal distribution


///////////////////////////////////////
// Wrapers for kernel functions.
///////////////////////////////////////

__global__ void gg_f1(float v, float *params, float *result){ *result = g_f1(v, params); }
__global__ void gg_f2(float v, float *params, float *result){ *result = g_f2(v, params); }
__global__ void gg_f3(float v, float *params, float *result){ *result = g_f3(v, params); }

__global__ void
gg_pdf_uniform (float lower, float upper, float x, float *result)
{
    *result = g_pdf_uniform(lower, upper, x);
}

__global__ void
gg_pdf_normal (float mean, float sd, float x, float *result)
{
    *result = g_pdf_normal(mean, sd, x);
}


///////////////////////////////////////
// Test functions' prototypes.
///////////////////////////////////////


void CallGF1(float v, float *params);
void CallGF2(float v, float *params);
void CallGF3(float v, float *params);

void CallGPdfUniform(float lower, float upper, float x);
void CallGPdfNormal(float mean, float sd, float x);



///////////////////////////////////////
// Main Test.
///////////////////////////////////////

int TestKernel()
{
    // Test for (f1, f2, f3)
    for (float i = 0; i < 10; i += 0.1) {
        float params[] = {1, 0.1, 0.1,
                          2, 0.1, 0.1};
        float v = i / 10.0f;

        CallGF1(v, params);
        CallGF2(v, params);
        CallGF3(v, params);
    }
    printf("Test for f1, f2, f3 passed.\n");


    // Test for g_pdf_uniform, g_pdf_normal
    for (float i = -100.0; i < 100.0; i += 1.0) {
        CallGPdfUniform(0.0, 1.0, i);
        CallGPdfUniform(-1.0, 1.0, i);
        CallGPdfUniform(-1.0, 0.0, i);
        CallGPdfUniform(0.0, 0.0, i);
        CallGPdfUniform(1.0, -1.0, i);

        CallGPdfNormal(0.0, 1.0, i);
        CallGPdfNormal(0.0, 1.0, i);
        CallGPdfNormal(0.0, 3.0, i);
        CallGPdfNormal(10.0, 3.0, i);
    }
    printf("Test for g_pdf_uniform, g_pdf_normal passed.\n");


    printf("TestKernel() passed!\n");
}



///////////////////////////////////////
// Definition of Test functions.
///////////////////////////////////////

void
CallGF1 (float v, float *params)
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
    checkCudaErrors(cudaMalloc((void**)&params_D, sizeof(float) * 6));
    checkCudaErrors(cudaMalloc((void**)&result_D, sizeof(float) * 1));

    checkCudaErrors(cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice));
    gg_f1<<< 1, 1 >>>(v, params_D, result_D);
    checkCudaErrors(cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost));

    NEAR(cpu, gpu);
}


void
CallGF2 (float v, float *params)
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
    checkCudaErrors(cudaMalloc((void**)&params_D, sizeof(float) * 6));
    checkCudaErrors(cudaMalloc((void**)&result_D, sizeof(float) * 1));

    checkCudaErrors(cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice));
    gg_f2<<< 1, 1 >>>(v, params_D, result_D);
    checkCudaErrors(cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost));

    NEAR(cpu, gpu);
}


void
CallGF3 (float v, float *params)
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
    checkCudaErrors(cudaMalloc((void**)&params_D, sizeof(float) * 6));
    checkCudaErrors(cudaMalloc((void**)&result_D, sizeof(float) * 1));

    checkCudaErrors(cudaMemcpy(params_D, params, sizeof(float) * 6, cudaMemcpyHostToDevice));
    gg_f3<<< 1, 1 >>>(v, params_D, result_D);
    checkCudaErrors(cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost));

    NEAR(cpu, gpu);
}


void
CallGPdfUniform (float lower, float upper, float x)
{
    float cpu, gpu;
    bool is_error = false;

    // CPU
    try {
        cpu = boost::math::pdf(boost::math::uniform_distribution<float>(lower, upper), x);
    } catch (...) {
        is_error = true;
    }

    // GPU
    float *result_D;
    checkCudaErrors(cudaMalloc((void**)&result_D, sizeof(float) * 1));
    gg_pdf_uniform<<< 1, 1 >>>(lower, upper, x, result_D);
    checkCudaErrors(cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost));

    if (is_error) {
        assert(gpu == 0.0f);
    }
    else {
        NEAR(cpu, gpu);
    }
}


void
CallGPdfNormal (float mean, float sd, float x)
{
    float cpu, gpu;
    bool is_error = false;

    // CPU
    try {
        cpu = boost::math::pdf(boost::math::normal_distribution< double >( mean, sd ), x);
    } catch (...) {
        is_error = true;
    }

    // GPU
    float *result_D;
    checkCudaErrors(cudaMalloc((void**)&result_D, sizeof(float) * 1));
    gg_pdf_normal<<< 1, 1 >>>(mean, sd, x, result_D);
    checkCudaErrors(cudaMemcpy(&gpu, result_D, sizeof(float), cudaMemcpyDeviceToHost));

    if (is_error) {
        assert(gpu == 0.0f);
    }
    else {
        NEAR(cpu, gpu);
    }
}
