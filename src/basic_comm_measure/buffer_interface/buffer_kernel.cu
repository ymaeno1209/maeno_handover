#include <stdio.h>
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

size_t findMax(size_t *array, int array_element);

__global__ void pack_data(char *dest_data, char *src_data, size_t src_data_offset_start, size_t packed_data_byte, size_t width_byte, size_t square_byte)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // idxがpacked_data_byteより小さい場合に処理
    if (idx < packed_data_byte)
    {
        size_t height_count = (size_t)idx / width_byte;
        size_t vertical_address_shift = square_byte * height_count;
        size_t horizontal_address_shift = (size_t)idx % width_byte;
        size_t src_data_index = src_data_offset_start + horizontal_address_shift - vertical_address_shift;
        int dest_data_index = idx;

        // デバッグ用の出力
        // printf("cuda :idx=%d, src_data_index=%lu, dest_data_index=%lu,horizontal_address_shift=%lu,height_count=%lu,vertical_address_shift=%lu,square_byte=%lu\n", idx, src_data_index, dest_data_index, horizontal_address_shift, height_count, vertical_address_shift, square_byte);

        dest_data[dest_data_index] = src_data[src_data_index];
    }
}

__global__ void con_pack_data(char *dest_data, char *src_data, size_t *src_data_offset_start, size_t *packed_data_byte, size_t *width_byte, size_t square_byte, int con_packing_count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t height_count;
    size_t vertical_address_shift;
    size_t horizontal_address_shift;
    size_t src_data_index;
    int dest_data_index = 0;
    // idxがpacked_data_byteより小さい場合に処理
    for (int i = 0; i < con_packing_count; i++)
    {
        if (idx < packed_data_byte[i])
        {
            height_count = (size_t)idx / width_byte[i];
            vertical_address_shift = square_byte * height_count;
            horizontal_address_shift = (size_t)idx % width_byte[i];
            src_data_index = src_data_offset_start[i] + horizontal_address_shift - vertical_address_shift;
            dest_data_index = idx + i * packed_data_byte[i];

            // デバッグ用の出力
            // printf("cuda :idx=%d, src_data_index=%lu, dest_data_index=%lu,horizontal_address_shift=%lu,height_count=%lu,vertical_address_shift=%lu,square_byte=%lu\n", idx, src_data_index, dest_data_index, horizontal_address_shift, height_count, vertical_address_shift, square_byte);

            dest_data[dest_data_index] = src_data[src_data_index];
        }
    }
}

extern "C" void
call_pack_data(char *dest_data, char *src_data, size_t src_data_offset_start, size_t packed_data_byte, size_t width_byte, size_t square_byte, cudaStream_t packing_stream)
{
    size_t block_size = 256; // 256が1ブロックの最大
    dim3 block(block_size);

    size_t blocksPerGrid = (packed_data_byte + block_size - 1L) / block_size;
    dim3 grid(blocksPerGrid);
    pack_data<<<grid, block, 0, packing_stream>>>(dest_data, src_data, src_data_offset_start, packed_data_byte, width_byte, square_byte);
}

extern "C" void
call_con_pack_data(char *dest_data, char *src_data, size_t *src_data_offset_start, size_t *packed_data_byte, size_t packed_data_byte_max, size_t *width_byte, size_t square_byte, int con_packing_count, cudaStream_t packing_stream)
{
    size_t block_size = 256; // 256が1ブロックの最大
    dim3 block(block_size);
    size_t blocksPerGrid = (packed_data_byte_max + block_size - 1L) / block_size;
    dim3 grid(blocksPerGrid);
    con_pack_data<<<grid, block, 0, packing_stream>>>(dest_data, src_data, src_data_offset_start, packed_data_byte, width_byte, square_byte, con_packing_count);
}

size_t findMax(size_t *array, int array_element)
{
    size_t max = 0; // 配列に負の値も含まれる可能性を考慮して最小値を初期値に設定

    for (int i = 0; i < array_element; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
        }
    }
    return max;
}