extern "C" __global__ void
complexMagnitude(float* __restrict__ dst, float*__restrict__ src, int N){

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        int e = 2*i;
        dst[i] = sqrtf(src[e]*src[e] + src[e+1] * src[e+1]);
    }
}