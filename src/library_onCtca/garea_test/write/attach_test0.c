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
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        int progid, gareaid0, gareaid1, gareaid2;
        int intparams[2];
        int garea_params[1];
        size_t data_element = 5;
        size_t data_size = data_element * sizeof(int);
        int *data_D;
        int *data_H;
        int i, j, k;
        int prognum = 1;
        int intparams_num = 2;
        int target_world_rank;

        cudaMalloc((void **)&data_D, data_size);
        data_H = (int *)malloc(data_size);

        printf("req%d: garea_create do\n", myrank);
        CTCAR_garea_create(&gareaid0);
        printf("req%d: garea_create done\n", myrank);

        cudaMemset(data_D, 0, data_size);

        printf("req%d: garea_attach do\n", myrank);
        CTCAR_garea_attach(gareaid0, data_D, data_size);
        printf("req%d: garea_attach done\n", myrank);
        MPI_Barrier(MPI_COMM_WORLD);

        cudaMemcpy(data_H, data_D, data_size, cudaMemcpyDeviceToHost);
        for (i = 0; i < data_element; i++)
        {
            printf("req%d: data_H[%d]=%d\n", myrank, i, data_H[i]);
        }

        CTCAR_garea_detach(gareaid0, data_D);
        printf("req%d garea_detach done\n", myrank);

        CTCAR_garea_delete();
        printf("req%d: garea_delete done\n", myrank);
        cudaFree(data_D);
    }
    else if (myrank == 1)
    {
        int progid, fromrank, gareaid0, gareaid1, gareaid2;
        int intparams[2];
        int reqinfo[CTCAC_REQINFOITEMS]; // reqinf[4]
        int data[6 * 400];
        int intparams_num = 2;
        int garea_params[1];
        int garea_params_num = 1;

        printf("cpr%d: garea_create do\n", myrank);
        CTCAC_garea_create(&gareaid0);
        printf("cpr%d: garea_create done\n", myrank);

        // garea_write debug
        MPI_Barrier(MPI_COMM_WORLD);

        CTCAC_garea_delete();

        printf("cpr%d: finalize\n", myrank);
    }
    else
    {
        int fromrank, gareaid0, gareaid1, gareaid2;
        int intparams[2];
        int garea_params[1];
        int *data;
        int c, datasize;
        int garea_params_num = 1;
        size_t data_element = 5;

        data = (int *)malloc(data_element * sizeof(int));

        printf("wrk%d: garea_create do\n", myrank);
        CTCAW_garea_create(&gareaid0);
        printf("wrk%d: garea_create done\n", myrank);

        int target_world_rank = 0;
        size_t offset = 0;

        for (int i = 0; i < data_element; i++)
        {
            data[i] = 1000 + i;
        }

        printf("wrk%d: garea_write do\n", myrank);
        CTCAW_garea_write_int(gareaid0, target_world_rank, offset, data_element, data);
        printf("wrk%d: garea_write done\n", myrank);
        MPI_Barrier(MPI_COMM_WORLD);

        CTCAW_garea_delete();
        printf("wrk%d: garea_delete done\n", myrank);
    }
    MPI_Finalize();
}
