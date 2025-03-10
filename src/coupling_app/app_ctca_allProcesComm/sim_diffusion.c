#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/time.h>
#include <limits.h>
#include <sys/types.h>
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef OLD
double diffusion3d(int nprocs, int rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn)
{
    const float ce = kappa * dt / (dx * dx);
    const float cw = ce;
    const float cn = kappa * dt / (dy * dy);
    const float cs = cn;
    const float ct = kappa * dt / (dz * dz);
    const float cb = ct;

    const float cc = 1.0 - (ce + cw + cn + cs + ct + cb);

    int k = 0;
#pragma acc kernels async(0) present(f, fn)
#pragma acc loop independent collapse(2)
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            const int ix = nx * ny * (k + mgn) + nx * j + i;
            const int ip = i == nx - 1 ? ix : ix + 1;
            const int im = i == 0 ? ix : ix - 1;
            const int jp = j == ny - 1 ? ix : ix + nx;
            const int jm = j == 0 ? ix : ix - nx;
            const int kp = ix + nx * ny;
            const int km = (rank == 0 && k == 0) ? ix : ix - nx * ny;

            fn[ix] = cc * f[ix] + ce * f[ip] + cw * f[im] + cn * f[jp] + cs * f[jm] + ct * f[kp] + cb * f[km];
        }
    }
    k = nz - 1;
#pragma acc kernels async(1) present(f, fn)
#pragma acc loop independent collapse(2)
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            const int ix = nx * ny * (k + mgn) + nx * j + i;
            const int ip = i == nx - 1 ? ix : ix + 1;
            const int im = i == 0 ? ix : ix - 1;
            const int jp = j == ny - 1 ? ix : ix + nx;
            const int jm = j == 0 ? ix : ix - nx;
            const int kp = (rank == nprocs - 1 && k == nz - 1) ? ix : ix + nx * ny;
            const int km = ix - nx * ny;

            fn[ix] = cc * f[ix] + ce * f[ip] + cw * f[im] + cn * f[jp] + cs * f[jm] + ct * f[kp] + cb * f[km];
        }
    }

#pragma acc kernels async(2) present(f, fn)
#pragma acc loop independent collapse(3)
    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                const int ix = nx * ny * (k + mgn) + nx * j + i;
                const int ip = i == nx - 1 ? ix : ix + 1;
                const int im = i == 0 ? ix : ix - 1;
                const int jp = j == ny - 1 ? ix : ix + nx;
                const int jm = j == 0 ? ix : ix - nx;
                const int kp = ix + nx * ny;
                const int km = ix - nx * ny;

                fn[ix] = cc * f[ix] + ce * f[ip] + cw * f[im] + cn * f[jp] + cs * f[jm] + ct * f[kp] + cb * f[km];
            }
        }
    }

    return (double)(nx * ny * nz) * 13.0;
}
#endif

void init(int nprocs, int rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float *f)
{
    const float kx = 2.0 * M_PI;
    const float ky = kx;
    const float kz = kx;
    int i, j, k;

    for (k = -mgn; k < nz + mgn; k++)
    {
        for (j = 0; j < ny; j++)
        {
            for (i = 0; i < nx; i++)
            {
                const int ix = nx * ny * (k + mgn) + nx * j + i;
                const float x = dx * ((float)i + 0.5);
                const float y = dy * ((float)j + 0.5);
                const float z = dz * ((float)(k + nz * rank) + 0.5);

                f[ix] = 0.125 * (1.0 - cos(kx * x)) * (1.0 - cos(ky * y)) * (1.0 - cos(kz * z));
                // cudaMemset(&(f[ix]),0.125*(1.0 - cos(kx*x))*(1.0 - cos(ky*y))*(1.0 - cos(kz*z)),1);
            }
        }
    }
}
void init_inc(int num, int nprocs, int rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float *f)
{
    int i, j, k;
    for (i = 0; i < nz; i++)
    {
        for (j = 0; j < ny; j++)
        {
            for (k = 0; k < nx; k++)
            {
                int cell = (i + 1) * nx * ny + j * nx + k;
                f[cell] = rank * nx * nz + i * nx + j * num * num + k;
            }
        }
    }
}

double err(int nprocs, int rank, double time, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float kappa, const float *f)
{
    const float kx = 2.0 * M_PI;
    const float ky = kx;
    const float kz = kx;

    const float ax = exp(-kappa * time * (kx * kx));
    const float ay = exp(-kappa * time * (ky * ky));
    const float az = exp(-kappa * time * (kz * kz));

    double ferr = 0.0;
    int i, j, k;

    for (k = 0; k < nz; k++)
    {
        for (j = 0; j < ny; j++)
        {
            for (i = 0; i < nx; i++)
            {
                const int ix = nx * ny * (k + mgn) + nx * j + i;
                const float x = dx * ((float)i + 0.5);
                const float y = dy * ((float)j + 0.5);
                const float z = dz * ((float)(k + nz * rank) + 0.5);

                const float f0 = 0.125 * (1.0 - ax * cos(kx * x)) * (1.0 - ay * cos(ky * y)) * (1.0 - az * cos(kz * z));

                ferr += (f[ix] - f0) * (f[ix] - f0);
            }
        }
    }

    return ferr;
}
