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
    char *create_data;
    MPI_Win win;
    int datasize = 1;

    // MPIの初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ローカルデータとリモートデータのメモリ割り当て

    cudaMalloc((void **)&create_data, datasize);
    cudaMemset(create_data, 'a', datasize);

    // MPIウィンドウの作成
    printf("rank=%d, win_create do\n", rank);
    MPI_Win_create(create_data, datasize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    printf("rank=%d, win_create done\n", rank);

    MPI_Win_free(&win);
    cudaFree(create_data);

    MPI_Finalize();

    return 0;
}
