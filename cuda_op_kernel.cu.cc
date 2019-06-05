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


__global__ void AddOneKernel(const float* d_array, const int elements, float* d_max) {

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
    d_max[gid] = shared[tid];
    if (tid == -1){
      atomicMaxf(d_max, shared[0]);
//      atomicMaxf(d_max+1, shared[1]);
//      atomicMaxf(d_max+2, shared[2]);
//      d_max[0] = shared[0];
    }

}

void AddOneKernelLauncher(const float* in, const int N, float* out) {
  AddOneKernel<<<32, 256, 256*4>>>(in, N, out);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}