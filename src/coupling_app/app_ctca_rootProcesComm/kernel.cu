#include <stdio.h>
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include "sim_misc.h"
#define blockDim_x 32
#define blockDim_y 8
#define blockDim_z 8

__global__ void top_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn)
{
    const float ce = kappa * dt / (dx * dx);
    const float cw = ce;
    const float cn = kappa * dt / (dy * dy);
    const float cs = cn;
    const float ct = kappa * dt / (dz * dz);
    const float cb = ct;
    const float cc = 1.0 - (ce + cw + cn + cs + ct + cb);
    float f0, f1, f2;
    int gridDim_x = nx / blockDim.x;
    int gridDim_y = ny / blockDim.y;
    int i = blockIdx.x % gridDim_x;
    int j = (blockIdx.x / gridDim_x) % gridDim_y;
    int sharedDim_x = blockDim.x + 2;

    int f_place = nx * ny * nz + (i * blockDim.x + threadIdx.x) + (j * blockDim.y + threadIdx.y) * nx;
    int s_place = 1 + threadIdx.x + (threadIdx.y + 1) * sharedDim_x;
    f1 = f[f_place];
    if (new_rank == new_nprocs - 1)
        f0 = f1;
    else
        f0 = f[f_place + nx * ny];
    f2 = f[f_place - nx * ny];

    extern __shared__ float sharedData[];
    sharedData[s_place] = f1;

    // front
    if (threadIdx.y == 0)
    {
        if (j == 0)
            sharedData[s_place - sharedDim_x] = f1;
        else
            sharedData[s_place - sharedDim_x] = f[f_place - nx];
    }
    // back
    if (threadIdx.y == blockDim_y - 1)
    {
        if (j == gridDim_y - 1)
            sharedData[s_place + sharedDim_x] = f1;
        else
            sharedData[s_place + sharedDim_x] = f[f_place + nx];
    }
    // left
    if (threadIdx.x == 0)
    {
        if (i == 0)
            sharedData[s_place - 1] = f1;
        else
            sharedData[s_place - 1] = f[f_place - 1];
    }
    // right
    if (threadIdx.x == blockDim_x - 1)
    {
        if (i == gridDim_x - 1)
            sharedData[s_place + 1] = f1;
        else
            sharedData[s_place + 1] = f[f_place + 1];
    }

    __syncthreads();

    fn[f_place] = cc * f1 + ce * sharedData[s_place + 1] + cw * sharedData[s_place - 1] + cn * sharedData[s_place + sharedDim_x] + cs * sharedData[s_place - sharedDim_x] + ct * f2 + cb * f0;
}

__global__ void bot_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn)
{
    const float ce = kappa * dt / (dx * dx);
    const float cw = ce;
    const float cn = kappa * dt / (dy * dy);
    const float cs = cn;
    const float ct = kappa * dt / (dz * dz);
    const float cb = ct;
    const float cc = 1.0 - (ce + cw + cn + cs + ct + cb);
    float f0, f1, f2;
    int gridDim_x = nx / blockDim_x;
    int gridDim_y = ny / blockDim_y;
    int i = blockIdx.x % gridDim_x;
    int j = (blockIdx.x / gridDim_x) % gridDim_y;
    int sharedDim_x = blockDim.x + 2;
    int f_place = nx * ny + (i * blockDim_x + threadIdx.x) + (j * blockDim_y + threadIdx.y) * nx;
    int s_place = 1 + threadIdx.x + (threadIdx.y + 1) * sharedDim_x;
    f1 = f[f_place];
    f0 = f[f_place + nx * ny];
    if (new_rank == 0)
        f2 = f1;
    else
        f2 = f[f_place - nx * ny];

    extern __shared__ float sharedData[];
    sharedData[s_place] = f1;

    // front
    if (threadIdx.y == 0)
    {
        if (j == 0)
            sharedData[s_place - sharedDim_x] = f1;
        else
            sharedData[s_place - sharedDim_x] = f[f_place - nx];
    }
    // back
    if (threadIdx.y == blockDim_y - 1)
    {
        if (j == gridDim_y - 1)
            sharedData[s_place + sharedDim_x] = f1;
        else
            sharedData[s_place + sharedDim_x] = f[f_place + nx];
    }
    // left
    if (threadIdx.x == 0)
    {
        if (i == 0)
            sharedData[s_place - 1] = f1;
        else
            sharedData[s_place - 1] = f[f_place - 1];
    }
    // right
    if (threadIdx.x == blockDim_x - 1)
    {
        if (i == gridDim_x - 1)
            sharedData[s_place + 1] = f1;
        else
            sharedData[s_place + 1] = f[f_place + 1];
    }

    __syncthreads();

    fn[f_place] = cc * f1 + ce * sharedData[s_place + 1] + cw * sharedData[s_place - 1] + cn * sharedData[s_place + sharedDim_x] + cs * sharedData[s_place - sharedDim_x] + ct * f0 + cb * f2;
}
__global__ void mid_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn)
{
    const float ce = kappa * dt / (dx * dx);
    const float cw = ce;
    const float cn = kappa * dt / (dy * dy);
    const float cs = cn;
    const float ct = kappa * dt / (dz * dz);
    const float cb = ct;
    const float cc = 1.0 - (ce + cw + cn + cs + ct + cb);
    float f0, f1, f2;
    int l;
    int gridDim_x = nx / blockDim_x;
    int gridDim_y = ny / blockDim_y;
    int gridDim_z = nz / blockDim_z;
    int i = blockIdx.x % gridDim_x;
    int j = (blockIdx.x / gridDim_x) % gridDim_y;
    int k = blockIdx.x / (gridDim_x * gridDim_y);
    int sharedDim_x = blockDim.x + 2;
    int f_place = 2 * nx * ny + (i * blockDim_x + threadIdx.x) + (j * nx * blockDim_y + threadIdx.y * nx) + k * nx * ny * blockDim_z;
    int s_place = 1 + threadIdx.x + (threadIdx.y + 1) * sharedDim_x;
    f1 = f[f_place];
    f2 = f[f_place - nx * ny];
    extern __shared__ float sharedData[];
    // #pragma unroll
    for (l = 0; l < blockDim_z; l++)
    {
        f0 = f[f_place + nx * ny];
        sharedData[s_place] = f1;
        // front
        if (threadIdx.y == 0)
        {
            if (j == 0)
                sharedData[s_place - sharedDim_x] = f1;
            else
                sharedData[s_place - sharedDim_x] = f[f_place - nx];
        }
        // back
        if (threadIdx.y == blockDim_y - 1)
        {
            if (j == gridDim_y - 1)
                sharedData[s_place + sharedDim_x] = f1;
            else
                sharedData[s_place + sharedDim_x] = f[f_place + nx];
        }
        // left
        if (threadIdx.x == 0)
        {
            if (i == 0)
                sharedData[s_place - 1] = f1;
            else
                sharedData[s_place - 1] = f[f_place - 1];
        }
        // right
        if (threadIdx.x == blockDim_x - 1)
        {
            if (i == gridDim_x - 1)
                sharedData[s_place + 1] = f1;
            else
                sharedData[s_place + 1] = f[f_place + 1];
        }
        __syncthreads();
        fn[f_place] = cc * f1 + ce * sharedData[s_place + 1] + cw * sharedData[s_place - 1] + cn * sharedData[s_place + sharedDim_x] + cs * sharedData[s_place - sharedDim_x] + ct * f0 + cb * f2;
        f2 = f1;
        f1 = f0;
        f_place += nx * ny;
        if (k == gridDim_z - 1 && l == blockDim_z - 3)
            break;
    }
}
// GPU上にあるデータをGPUのメモリへコピー
__global__ void copy_data(float *dest, float *src, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size)
    {
        dest[idx] = src[idx]; // src から dest へコピー
    }
}
__global__ void packing_data(float *dest, float *src, int copy_element, int src_start_element, int dest_stride_element, int src_stride_element, int loop)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < copy_element)
    {
        for (int i = 0; i < loop; i++)
        {
            dest[idx + i * copy_element] = src[src_start_element + idx + i * src_stride_element];
        }
    }
}
__global__ void con_packing_data(float *dest_data, float *src_data, int num)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t src_addr;
    size_t dest_addr;
    // int idx_x = idx % num;
    // int idx_y = idx / num;
    if (idx < num * num)
    {
        for (int i = 0; i < num; i++)
        {
            src_addr = num * num * (num - 1) + idx - i * num * num;
            dest_addr = idx + i * num * num;
            dest_data[dest_addr] = src_data[src_addr];
        }
    }
}
__global__ void con_packing_data_new_rootrank(float *dest_data, float *src_data, int num, int con_packing_count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t src_addr;
    size_t dest_addr;

    if (idx < num * num)
    {
        for (int i; i < con_packing_count; i++)
        {

            src_addr = idx - i * num * num;
            dest_addr = idx + i * num * num;
            dest_data[dest_addr] = src_data[src_addr];
        }
    }
}

__global__ void con_packing_data_new(float *dest_data, float *src_data, int num, int con_packing_count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t src_addr;
    size_t dest_addr;

    if (idx < num * num)
    {
        for (int i = 0; i < num; i++)
        {
            src_addr = num * num * (num - 1) + idx - i * num * num;
            dest_addr = idx + i * num * num;
            dest_data[dest_addr] = src_data[src_addr];
        }
    }
}
__global__ void
init_kernel(int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float *f)
{
    const float kx = 2.0f * M_PI;
    const float ky = kx;
    const float kz = kx;

    // 3Dグリッド、ブロック内のスレッドIDを取得
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z - mgn; // マージンを考慮

    if (i < nx && j < ny && k >= -mgn && k < nz + mgn)
    {
        int ix = nx * ny * (k + mgn) + nx * j + i;
        float x = dx * (i + 0.5f);
        float y = dy * (j + 0.5f);
        float z = dz * ((k + nz * new_rank) + 0.5f);

        f[ix] = 0.125f * (1.0f - cosf(kx * x)) * (1.0f - cosf(ky * y)) * (1.0f - cosf(kz * z));
    }
}

extern "C" void
call_top_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t top_stream)
{
    int blockDimx = blockDim_x;
    int blockDimy = blockDim_y;
    int blockDimz = 1;
    dim3 block(blockDimx, blockDimy, blockDimz);
    int blocksPerGrid = (nx / blockDimx) * (ny / blockDimy);
    dim3 grid(blocksPerGrid, 1);
    int sharedMemSize = (blockDimx + 2) * (blockDimy + 2) * (blockDimz + 2) * sizeof(float);
    top_stencil<<<grid, block, sharedMemSize, top_stream>>>(calc, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn);
}

extern "C" void
call_bot_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t bot_stream)
{
    int blockDimx = blockDim_x;
    int blockDimy = blockDim_y;
    int blockDimz = 1;
    dim3 block(blockDimx, blockDimy, blockDimz);
    int blocksPerGrid = (nx / blockDimx) * (ny / blockDimy);
    dim3 grid(blocksPerGrid, 1);
    int sharedMemSize = (blockDimx + 2) * (blockDimy + 2) * (blockDimz + 2) * sizeof(float);
    bot_stencil<<<grid, block, sharedMemSize, bot_stream>>>(calc, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn);
}

extern "C" void
call_mid_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t mid_stream)
{
    int blockDimx = blockDim_x;
    int blockDimy = blockDim_y;
    int blockDimz = 1;
    dim3 block(blockDimx, blockDimy, blockDimz);
    int blocksPerGrid = (nx / blockDim_x) * (ny / blockDim_y) * (nz / blockDim_z);
    dim3 grid(blocksPerGrid, 1);
    int sharedMemSize = (blockDimx + 2) * (blockDimy + 2) * sizeof(float);
    mid_stencil<<<grid, block, sharedMemSize, mid_stream>>>(calc, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn);
}

extern "C" void
call_copy_data(float *dest, float *src, int size)
{
    int block_size = 256;
    dim3 block(block_size);
    int blocksPerGrid = (size + block_size - 1) / block_size;
    dim3 grid(blocksPerGrid);
    copy_data<<<grid, block>>>(dest, src, size);
    cudaDeviceSynchronize();
}
extern "C" void
call_packing_data(float *dest, float *src, int copy_element, int src_start_element, int dest_stride_element, int src_stride_element, int loop, cudaStream_t packing_stream)
{
    int block_size = 256;
    dim3 block(block_size);
    int blocksPerGrid = (copy_element + block_size - 1) / block_size;
    dim3 grid(blocksPerGrid);
    packing_data<<<grid, block, 0, packing_stream>>>(dest, src, copy_element, src_start_element, dest_stride_element, src_stride_element, loop);
}
extern "C" void
call_con_packing_data(float *dest_data, float *src_data, int num, cudaStream_t con_packing_stream)
{
    int block_size = 256;
    dim3 block(block_size);
    int blocksPerGrid = (num * num + block_size - 1) / block_size;
    dim3 grid(blocksPerGrid);
    con_packing_data<<<grid, block, 0, con_packing_stream>>>(dest_data, src_data, num);
}

extern "C" void
call_kernel_init(int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float *f)
{
    // グリッドとブロックサイズを設定
    dim3 threadsPerBlock(8, 8, 8); // スレッド数は適宜調整可能
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + 2 * mgn + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // CUDAカーネルを呼び出す
    init_kernel<<<numBlocks, threadsPerBlock>>>(new_rank, nx, ny, nz, mgn, dx, dy, dz, f);
    cudaDeviceSynchronize();
}
