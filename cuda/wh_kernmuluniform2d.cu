extern "C" __global__ void
kernmulUniform2D(float* __restrict__ Fxx, float* __restrict__ Fyy, float* __restrict__ Fzz,
                 float* __restrict__ Fxy,
                 float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKzz,
                 float* __restrict__  fftKxy,
                 float* __restrict__ ftu, int Nx, int Ny, int Nz){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < Nx && iy < Ny) {
        int I = iy*Nx + ix;

        int e = 2 * I;
        float reftu = ftu[e];       float imftu = ftu[e+1];

        // symmetry factor
        float fxy = 1.0f;
        if (iy > Ny/2) {
            iy = Ny-iy;
            fxy = -fxy;
        }
        I = iy*Nx + ix;

        float Kxx = fftKxx[I];
        float Kyy = fftKyy[I];
        float Kzz = fftKzz[I];
        float Kxy = fxy * fftKxy[I];

        Fxx[e] = Kxx * reftu;       Fxx[e+1] = Kxx * imftu;  
        Fxy[e] = Kxy * reftu;       Fxy[e+1] = Kxy * imftu;
        Fyy[e] = Kyy * reftu;       Fyy[e+1] = Kyy * imftu;
        Fzz[e] = Kzz * reftu;       Fzz[e+1] = Kzz * imftu;


    }


}