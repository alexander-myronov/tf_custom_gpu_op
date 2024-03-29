#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <float.h>


__device__ void atomicMax(float* const address, const float value)
{
	if (*address >= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI, assumed;

	do
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
	} while (assumed != old);
}

__inline__ __device__ float warpReduceMax(float val)
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }

    return val;
}

template<class T>
__inline__ __device__ float warpBroadcast(T val, int predicate)
{
    const unsigned int FULL_MASK = 0xffffffff;

    unsigned int mask = __ballot_sync(FULL_MASK, predicate);

    int lane = 0;
    for (;!(mask & 1); ++lane)
    {
        mask >>= 1;
    }

    return __shfl_sync(FULL_MASK, val, lane);
}

__device__ void reduceMaxIdxOptimizedWarpShared(const float* __restrict__ input,
                const int size, float* maxOut,
                int* maxIdxOut,
                float ignore)
{
    __shared__ float sharedMax1;
    __shared__ int sharedMaxIdx1;


    if (0 == threadIdx.x)
    {
        sharedMax1 = 0.0f;
        sharedMaxIdx1  = 0;
    }

    __syncthreads();

    float localMax1 = 0.0f;
    int localMaxIdx1 = 0;


    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];
//        float cap =
        if (localMax1 < val && val < ignore)
        {
            localMax1 = val;
            localMaxIdx1 = i;
        }
    }

    const float warpMax1 = warpReduceMax(localMax1);


    const int warpMaxXY1 = warpBroadcast(localMaxIdx1, warpMax1 == localMax1);


    const int lane = threadIdx.x % warpSize;

    if (lane == 0)
    {
        atomicMax(&sharedMax1, warpMax1);
    }

    __syncthreads();

    if (lane == 0)
    {
        if (sharedMax1 == warpMax1)
        {
            sharedMaxIdx1 = warpMaxXY1;
        }
    }

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax1;
        *maxIdxOut = sharedMaxIdx1;
    }
}


__global__ void MaxKernel(const float* d_array,  float* d_max, float* d_max_init,
                                        const int batch_size,
                                        const int seq_len,
                                        const int n_features,
                                        const int k) {

    int index;

    for (int sample = 0; sample < batch_size; sample++){
        for (int feature = 0; feature < n_features; feature++){
            reduceMaxIdxOptimizedWarpShared(
                    d_array + sample * n_features * seq_len + feature * seq_len,
                    seq_len,
                    d_max+sample*n_features + feature,
                     &index,
                    FLT_MAX );
                __syncthreads();




        }

    }

    __syncthreads();
    return;
}

void MaxKernelLauncher(const float* in, float* out, float *max_init, const int batch_size, const int seq_len, const int n_features)
{

    MaxKernel<<<12, 1024>>>(in, out, max_init, batch_size, seq_len, n_features, 1);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}