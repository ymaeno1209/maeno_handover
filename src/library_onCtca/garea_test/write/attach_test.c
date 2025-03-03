#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include "ctca.h"
#include "../garea.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define GAREA_BASE_ADDRESSES_INITIAL -1

int main()
{
    int myrank, nprocs;
    int num_gareas = 10;
    int gareaidctr;
    size_t garea_base_addresses_byte;
    volatile MPI_Aint *garea_base_addresses;
    MPI_Win win_garea_base_addresses;
    MPI_Win *win_garea_table;
    MPI_Aint base_toshare;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    size_t data_element = 5;
    size_t data_size = data_element * sizeof(int);
    int *data_D;
    int *data_H;

    cudaMalloc((void **)&data_D, data_size);
    data_H = (int *)malloc(data_size);

    printf("req%d: garea_create do\n", myrank);
    gareaidctr = 0;
    garea_base_addresses_byte = num_gareas * sizeof(MPI_Aint);
    garea_base_addresses = (MPI_Aint *)malloc(garea_base_addresses_byte);
    for (int i = 0; i < num_gareas; i++)
        garea_base_addresses[i] = (MPI_Aint)GAREA_BASE_ADDRESSES_INITIAL;
    MPI_Win_create((void *)garea_base_addresses, nprocs, sizeof(MPI_Aint), MPI_INFO_NULL, MPI_COMM_WORLD, &win_garea_base_addresses);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_garea_base_addresses);
    win_garea_table = (MPI_Win *)malloc(num_gareas * sizeof(MPI_Win));

    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &(win_garea_table[gareaidctr]));
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_garea_table[gareaidctr]);

    printf("req%d: garea_create done\n", myrank);

    cudaMemset(data_D, 0, data_size);

    printf("req%d: garea_attach do\n", myrank);

    MPI_Get_address(data_D, &base_toshare);

    MPI_Win_attach(win_garea_table[gareaidctr], data_D, data_size);
    garea_base_addresses[gareaidctr] = base_toshare;

    MPI_Win_flush_all(win_garea_table[gareaidctr]);
    MPI_Win_flush_all(win_garea_base_addresses);
    printf("req%d: garea_attach done\n", myrank);

    MPI_Win_detach(win_garea_table[gareaidctr], data_D);
    garea_base_addresses[gareaidctr] = GAREA_BASE_ADDRESSES_INITIAL;

    MPI_Win_flush_all(win_garea_table[gareaidctr]);
    MPI_Win_flush_all(win_garea_base_addresses);
    printf("req%d garea_detach done\n", myrank);

    if (win_garea_base_addresses == MPI_WIN_NULL)
    {
        printf("error: %d win_garea_base_addresses is MPI_WIN_NULL\n", myrank);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Win_unlock_all(win_garea_base_addresses);
    MPI_Win_free(&win_garea_base_addresses);

    for (int i = 0; i < gareaidctr; i++)
    {
        int gareaid = i;
        if (win_garea_table[gareaid] == MPI_WIN_NULL)
        {
            printf("error: %d win_garea_table is MPI_WIN_NULL\n", myrank);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        MPI_Win_unlock_all(win_garea_table[gareaid]);
        MPI_Win_free(&win_garea_table[gareaid]);
    }

    free_common_tables();
    printf("req%d: garea_delete done\n", myrank);

    cudaFree(data_D);

    MPI_Finalize();
}
