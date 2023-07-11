extern "C" __global__ void
tensorFieldFactor(float* __restrict__  dst,
    float* __restrict__ ms, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        if (ms[i] == 0.0f) {
            dst[i] = 0.0f;
        } else {
            dst[i] = - 1.0f / ms[i];
        }
    }
}