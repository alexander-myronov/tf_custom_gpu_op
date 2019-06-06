#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <float.h>

__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

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
        sharedMax1 = -1.0f;
        sharedMaxIdx1  = 0;
    }

    __syncthreads();

    float localMax1 = -1.0f;
    int localMaxIdx1 = 0;


    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];

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


__global__ void AddOneKernel(const float* d_array, const int elements, float* d_max, int k) {

    float cap = FLT_MAX;
    int index;
    float sum = 0;
    for (int i = 0; i < k; i++){
        reduceMaxIdxOptimizedWarpShared(d_array, elements, d_max+i, &index, cap);
//        __syncthreads();
        cap = d_max[i];
        sum += d_max[i];
    }
    if(threadIdx.x == 0)
        d_max[0] = sum / k;
    return;
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLT_MAX;



    while (gid < elements) {
        shared[tid] = max(shared[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

//    atomicMaxf(d_max+tid, shared[tid]);
//    d_max[gid] = shared[tid];
    if (tid == 0){
      atomicMaxf(d_max, shared[0]);
//      atomicMaxf(d_max+1, shared[1]);
//      atomicMaxf(d_max+2, shared[2]);
//      d_max[0] = shared[0];
    }

}

void AddOneKernelLauncher(const float* in, const int N, float* out) {
  AddOneKernel<<<32, 256>>>(in, N, out, 1);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}