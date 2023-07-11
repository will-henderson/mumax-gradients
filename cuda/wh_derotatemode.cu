extern "C" __global__ void
derotateMode(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
               float* __restrict__  mx,  float* __restrict__  my,
               float* __restrict__  Rxx, float* __restrict__  Rxy, float* __restrict__  Rxz,
               float* __restrict__  Ryx, float* __restrict__  Ryy, float* __restrict__  Ryz,
               int N){

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {  
        dstx[i] = Rxx[i]*mx[i] + Ryx[i]*my[i];
        dsty[i] = Rxy[i]*mx[i] + Ryy[i]*my[i];
        dstz[i] = Rxz[i]*mx[i] + Ryz[i]*my[i];
    }  
    
}