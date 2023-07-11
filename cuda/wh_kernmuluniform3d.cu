extern "C" __global__ void
kernmulUniform3D(float* __restrict__ Fxx, float* __restrict__ Fyy, float* __restrict__ Fzz,
                 float* __restrict__ Fyz, float* __restrict__ Fxz, float* __restrict__ Fxy,
                 float* __restrict__  fftKxx, float* __restrict__  fftKyy, float* __restrict__  fftKzz,
                 float* __restrict__  fftKyz, float* __restrict__  fftKxz, float* __restrict__  fftKxy,
                 float* __restrict__ ftu, int Nx, int Ny, int Nz){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if(ix < Nx && iy < Ny && iz < Nz){
        int I = (iz*Ny + iy)*Nx + ix; 
        int e = 2 * I;
        float reftu = ftu[e];       float imftu = ftu[e+1];

         // fetch kernel

        // minus signs are added to some elements if
        // reconstructed from symmetry.
        float signYZ = 1.0f;
        float signXZ = 1.0f;
        float signXY = 1.0f;

        // use symmetry to fetch from redundant parts:
        // mirror index into first quadrant and set signs.
        if (iy > Ny/2) {
            iy = Ny-iy;
            signYZ = -signYZ;
            signXY = -signXY;
        }
        if (iz > Nz/2) {
            iz = Nz-iz;
            signYZ = -signYZ;
            signXZ = -signXZ;
        }

        // fetch kernel element from non-redundant part
        // and apply minus signs for mirrored parts.
        I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
        float Kxx = fftKxx[I];
        float Kyy = fftKyy[I];
        float Kzz = fftKzz[I];
        float Kyz = fftKyz[I] * signYZ;
        float Kxz = fftKxz[I] * signXZ;
        float Kxy = fftKxy[I] * signXY;

        Fxx[e] = Kxx * reftu;        Fxx[e+1] = Kxx * imftu; 
        Fxy[e] = Kxy * reftu;        Fxy[e+1] = Kxy * imftu;
        Fxz[e] = Kxz * reftu;        Fxz[e+1] = Kxz * imftu;
        Fyy[e] = Kyy * reftu;        Fyy[e+1] = Kyy * imftu;
        Fyz[e] = Kyz * reftu;        Fyz[e+1] = Kyz * imftu;
        Fzz[e] = Kzz * reftu;        Fzz[e+1] = Kzz * imftu;
    }
}