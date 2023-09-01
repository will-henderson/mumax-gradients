extern "C" __global__ void
fourierMode(float* __restrict__ dstReal, float* __restrict__ dstImag,
            float fx, float fy, float fz, int Nx, int Ny, int Nz){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz){
        int I = (iz*Ny + iy)*Nx + ix;

        float expon = fx * ix + fy * iz + fz * iz;
        dstReal[I] = cosf(expon);
        dstImag[I] = sinf(expon);  
    }        

}