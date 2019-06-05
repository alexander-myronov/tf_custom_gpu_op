#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void AddOneKernel(const float* in, const int N, float* out) {

  float max1=0, max2=0, max3=0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    {
        out[i] = in[i];
        if (in[i] > max1)
        {
            max1 = in[i];
        }
        else
        {
            if (in[i] > max2)
            {
                max2 = in[i];
            }
            else
            {
                if (in[i] > max3)
                {
                    max3 = in[i];
                }
            }
        }
    }
    //out[blockIdx.x * blockDim.x + threadIdx.x] = max1;
    //out[blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x] = max2;
    //out[blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x + blockDim.x * gridDim.x] = max3;
    if (threadIdx.x==0)
    {
        out[blockIdx.x * blockDim.x + threadIdx.x] = max1;//(max1+max2+max3)/3.0;
    }

}

void AddOneKernelLauncher(const float* in, const int N, float* out) {
  AddOneKernel<<<32, 256>>>(in, N, out);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}