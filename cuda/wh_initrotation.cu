
extern "C" __global__ void
initRotation(float* __restrict__  mx,  float* __restrict__  my,  float* __restrict__  mz,
               float* __restrict__  Rxx, float* __restrict__  Rxy, float* __restrict__  Rxz,
               float* __restrict__  Ryx, float* __restrict__  Ryy, float* __restrict__  Ryz,
               float* __restrict__  Rzx, float* __restrict__  Rzy, float* __restrict__  Rzz,
               int N){

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        float Cth = mz[i];
        float Sth = sqrtf(1 - mz[i]*mz[i]);
        float Cph = mx[i] / Sth;
        float Sph = my[i] / Sth;

        Rxx[i] = Cth*Cph; Rxy[i] = Cth*Sph; Rxz[i] = -Sth;
        Ryx[i] = -Sph;    Ryy[i] = Cph;     Ryz[i] = 0.0f;
        Rzx[i] = Sth*Cph; Rzy[i] = Sth*Sph; Rzz[i] = Cth;
    }
}
