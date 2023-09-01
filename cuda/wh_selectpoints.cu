#include <stdint.h>

extern "C" __global__ void 
selectPoints(float* __restrict__ dst, int nSamples, 
             float* __restrict__ src, int Nx, int Ny, int Nz, 
             int32_t* __restrict__ sampleX, int32_t* __restrict__ sampleY, int32_t* __restrict__ sampleZ){

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < nSamples) {
        int e = 2 * i;

        int srcPoint = (sampleZ[i] * Ny + sampleY[i]) * Nx + sampleX[i];
        int srcPointe = 2 * srcPoint;

        dst[e] = src[srcPointe];
        dst[e+1] = src[srcPointe + 1];
    }
}