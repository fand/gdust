#pragma once

//#include <cuda.h>
//#include <curand.h>

//#include <cassert>



__device__ bool check_uniform_lower(float lower, float *result);
__device__ bool check_uniform_upper(float upper, float *result);
__device__ bool check_uniform(float lower, float upper, float *result);
__device__ bool check_uniform_x(float const& x, float *result);    
__device__ bool check_location(float location, float * result);
__device__ bool check_x(float x, float *result);
__device__ bool check_scale(float scale, float *result);
__device__ bool verify_lambda(float l, float *result);
__device__ bool verify_exp_x(float x, float *result);

__device__ float pdf_uniform( float lower, float upper, float x );
__device__ float pdf_normal( float mean, float sd, float x );

__device__ float myPDF( int distribution, float mean, float stddev, float v );

__device__ float f1 ( float v, void *params );
__device__ float f2 ( float v, void *params );
__device__ float f3 ( float z, void *params );
__device__ float f4 ( float k, void *params );

__global__ void integrate_kernel( void *params, int fnum, float *input_array, float *output_array, float range_min, float range_max );



// SOTA Parallel reduction

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};


/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
       T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch

    switch (threads)
    {
    case 512:
        reduce6<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
        reduce6<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
        reduce6<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
        reduce6<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
        reduce6<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
        reduce6<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
        reduce6<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
        reduce6<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
        reduce6<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
        reduce6<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }

}
