extern "C" __global__ void
addMul(float* __restrict__  dst,
      float* __restrict__  src1, float* src2, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        dst[i] += src1[i]*src2[i];
    }
}