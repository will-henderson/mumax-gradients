#include "amul.h"
#include "float3.h"
#include <stdint.h>

extern "C" __global__ void
mProp(float* __restrict__ new_lmx, float* __restrict__ new_lmy, float* __restrict__ new_lmz, 
    float* __restrict__ new_lbx, float* __restrict__ new_lby, float* __restrict__ new_lbz,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
    float* __restrict__ old_lmx, float* __restrict__ old_lmy, float* __restrict__ old_lmz,
    float* __restrict__ alpha_, float alpha_mul, int N) {

        int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
        if (i < N) {
            float3 m = {mx[i], my[i], mz[i]};
            float3 B = {bx[i], by[i], bz[i]};
            float3 l = {old_lmx[i], old_lmy[i], old_lmz[i]};

            float3 lxB = cross(l, B);
            float3 mxB = cross(m, B);
            float3 lxm = cross(l, m);

            float alpha = amul(alpha_, alpha_mul, i);
            float gilb = 1.0f / (1.0f + alpha * alpha);
            float3 m_torque = gilb * (lxB + alpha * (cross(l, mxB) + cross(lxm, B)));
            float3 new_lb = - gilb * (lxm + alpha * cross(lxm, m));

            // we also add the identity because this is from time integration.
            float3 new_lm = m_torque + l;
            new_lmx[i] = new_lm.x;
            new_lmy[i] = new_lm.y;
            new_lmz[i] = new_lm.z;

            new_lbx[i] = new_lb.x;
            new_lby[i] = new_lb.y;
            new_lbz[i] = new_lb.z;

        }
    }