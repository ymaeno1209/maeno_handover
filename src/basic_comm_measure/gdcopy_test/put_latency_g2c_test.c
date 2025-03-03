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
    char *local_data, *remote_data;
    MPI_Win win;
    int offset = 0;
    int target_rank = 1;
    int assert = 0;
    int skip = 10;
    int loop = 100;

    // MPIの初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        printf("datasize[Byte], time[us]\n");
    }

    for (int datasize = 1; datasize <= 1024; datasize *= 2)
    {

        // ローカルデータとリモートデータのメモリ割り当て
        if (rank == 0)
        {
            cudaMalloc((void **)&local_data, datasize);
            cudaMalloc((void **)&remote_data, datasize);
            cudaMemset(local_data, 'a', datasize);
            cudaMemset(remote_data, 'b', datasize);
        }
        else
        {
            local_data = (char *)malloc(datasize);
            remote_data = (char *)malloc(datasize);
            memset(local_data, 'a', datasize);
            memset(remote_data, 'b', datasize);
        }

        // MPIウィンドウの作成
        MPI_Win_create(remote_data, datasize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // 計測開始
        MPI_Barrier(MPI_COMM_WORLD); // 全プロセスの同期

        // MPI_Putを使用してデータをリモートプロセスに送信
        if (rank == 0)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, target_rank, assert, win);
            for (int i = 0; i < loop + skip; i++)
            {
                if (i == skip)
                {
                    start_time = MPI_Wtime();
                }
                MPI_Put(local_data, datasize, MPI_CHAR, target_rank, offset, datasize, MPI_CHAR, win);
                MPI_Win_flush(target_rank, win);
            }
            end_time = MPI_Wtime();
            MPI_Win_unlock(target_rank, win);
        }

        // 計測終了

        elapsed_time = (end_time - start_time) / loop;

        // 結果を表示
        if (rank == 0)
        {
            printf("%d, %f\n", datasize, elapsed_time * 1000 * 1000);
        }

        // メモリの解放とウィンドウの閉じる
        MPI_Win_free(&win);
        if (rank == 0)
        {
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
