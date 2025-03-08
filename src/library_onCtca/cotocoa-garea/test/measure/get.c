#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    double start_time, end_time;
    double elapsed_time = 0.0;
    int *local_data, *remote_data;
    int *local_data_H, *remote_data_H;
    MPI_Win win;
    int offset = 0;
    int target_rank = 0;
    int assert = 0;
    int skip = 100;
    int loop = 10000;
    int max_element = 1024 * 1024 * 8;

    // MPIの初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        printf("data_byte[Byte], time[us]\n");
    }

    for (int data_element = 1; data_element <= max_element; data_element *= 2)
    {
        int data_byte = data_element * sizeof(int);

        // ローカルデータとリモートデータのメモリ割り当て
        if (rank == 0)
        {
            local_data_H = (int *)malloc(data_byte);
            remote_data_H = (int *)malloc(data_byte);
            cudaMalloc((void **)&local_data, data_byte);
            cudaMalloc((void **)&remote_data, data_byte);
            for (int i = 0; i < data_element; i++)
            {
                local_data_H[i] = 0;
                remote_data_H[i] = 1;
            }
            cudaMemcpy(local_data, local_data_H, data_byte, cudaMemcpyHostToDevice);
            cudaMemcpy(remote_data, remote_data_H, data_byte, cudaMemcpyHostToDevice);
        }
        else
        {
            local_data = (int *)malloc(data_byte);
            remote_data = (int *)malloc(data_byte);
            for (int i = 0; i < data_element; i++)
            {
                local_data[i] = 0;
                remote_data[i] = 1;
            }
        }

        // MPIウィンドウの作成
        MPI_Win_create(remote_data, data_byte, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

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

        if (rank == 1)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, target_rank, assert, win);
            for (int i = 0; i < loop + skip; i++)
            {
                if (i == skip)
                {
                    start_time = MPI_Wtime();
                }
                // MPI_Put(local_data, data_element, MPI_INT, target_rank, offset, data_element, MPI_INT, win);
                MPI_Get(local_data, data_element, MPI_INT, target_rank, offset, data_element, MPI_INT, win);
                MPI_Win_flush(target_rank, win);
            }
            end_time = MPI_Wtime();
            MPI_Win_unlock(target_rank, win);
        }

        // 計測終了

        elapsed_time = (end_time - start_time) / loop;
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

        if (rank == 1)
        {
            printf("%d, %f\n", data_byte, elapsed_time * 1000 * 1000);
        }

        // メモリの解放とウィンドウの閉じる
        MPI_Win_free(&win);
        if (rank == 0)
        {
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

    // MPIの終了
    MPI_Finalize();

    return 0;
}
