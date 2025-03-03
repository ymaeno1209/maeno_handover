#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include "ctca.h"
#include "buffer.h"

int main(int argc, char *argv[])
{
    int rank, size;
    double start_time, end_time;
    double elapsed_time = 0.0;
    int *local_data, *remote_data;
    int *local_data_H, *remote_data_H;
    int offset = 0;
    int target_rank = 1;
    int assert = 0;
    int skip = 100;
    int loop = 10000;
    int gareaid;
    int max_element = 1024 * 1024 * 8;
    int data_division[3];
    data_division[0] = 1;
    data_division[1] = 1;
    data_division[2] = 1;

    // MPIの初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        CTCAR_init();
    }
    else
    {
        int progid = 0;
        int procs_per_subcomm = 1;
        CTCAW_init(progid, procs_per_subcomm);
    }

    if (rank == 0)
    {
        printf("global_data_byte[Byte], time[us]\n");
    }

    for (int data_element = 1; data_element <= max_element; data_element *= 2)
    {
        int global_data_element = data_element * data_element * data_element;
        size_t global_data_byte = data_element * sizeof(int);

        // ローカルデータとリモートデータのメモリ割り当て
        if (rank == 0)
        {
            local_data_H = (int *)malloc(global_data_byte);
            remote_data_H = (int *)malloc(global_data_byte);
            cudaMalloc((void **)&local_data, global_data_byte);
            cudaMalloc((void **)&remote_data, global_data_byte);
            for (int i = 0; i < data_element; i++)
            {
                local_data_H[i] = 0;
                remote_data_H[i] = 1;
            }
            cudaMemcpy(local_data, local_data_H, global_data_byte, cudaMemcpyHostToDevice);
            cudaMemcpy(remote_data, remote_data_H, global_data_byte, cudaMemcpyHostToDevice);

            bool data_direction = true;
            printf("req%d: buffer_init_withint do\n", rank);
            requester_buffer_init_withint(data_division, global_data_element, data_direction);
            printf("req%d: buffer_init_withint done\n", rank);
        }
        else
        {
            int num;
            local_data = (int *)malloc(global_data_byte);
            remote_data = (int *)malloc(global_data_byte);
            for (int i = 0; i < data_element; i++)
            {
                local_data[i] = 0;
                remote_data[i] = 1;
            }
            target_side_type target_side = (target_side_type)ZX;
            printf("wrk%d: buffer_init_withint do\n", rank);
            worker_buffer_init_withint(data_division, target_side, &num);
            printf("wrk%d: buffer_init_withint done\n", rank);
        }

        // 計測開始
        MPI_Barrier(MPI_COMM_WORLD); // 全プロセスの同期
        // if (rank == 0)
        // {
        //     cudaMemcpy(local_data_H, local_data, data_byte, cudaMemcpyDeviceToHost);
        //     cudaMemcpy(remote_data_H, remote_data, data_byte, cudaMemcpyDeviceToHost);
        //     printf("before: rank=%d, data_byte=%d, local_data[0]=%d, remote_data[0]=%d\n", rank, data_byte, local_data_H[0], remote_data_H[0]);
        // }
        // else
        // {
        //     printf("before: rank=%d, data_byte=%d, local_data[0]=%d, remote_data[0]=%d\n", rank, data_byte, local_data[0], remote_data[0]);
        // }

        // MPI_Putを使用してデータをリモートプロセスに送信
        if (rank == 0)
        {
            for (int i = 0; i < loop + skip; i++)
            {
                if (i == skip)
                {
                    start_time = MPI_Wtime();
                }
                requester_buffer_write(local_data, i);
            }
            end_time = MPI_Wtime();
        }
        else
        {
            for (int i = 0; i < loop + skip; i++)
            {
                worker_buffer_read(remote_data, i);
            }
        }

        elapsed_time = (end_time - start_time) / loop;

        // 結果を表示
        if (rank == 0)
        {
            printf("%d, %f\n", global_data_byte, elapsed_time * 1000 * 1000);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // if (rank == 0)
        // {
        //     cudaMemcpy(local_data_H, local_data, data_byte, cudaMemcpyDeviceToHost);
        //     cudaMemcpy(remote_data_H, remote_data, data_byte, cudaMemcpyDeviceToHost);
        //     printf("after: rank=%d, data_byte=%d, local_data[0]=%d, remote_data[0]=%d\n", rank, data_byte, local_data_H[0], remote_data_H[0]);
        // }
        // else
        // {
        //     printf("after: rank=%d, data_byte=%d, local_data[0]=%d, remote_data[0]=%d\n", rank, data_byte, local_data[0], remote_data[0]);
        // }

        // メモリの解放とウィンドウの閉じる
        if (rank == 0)
        {
            requester_buffer_fin();
            free(remote_data_H);
            free(local_data_H);
            cudaFree(local_data);
            cudaFree(remote_data);
        }
        else
        {
            free(local_data);
            free(remote_data);
        }
    }
    MPI_Finalize();

    return 0;
}
