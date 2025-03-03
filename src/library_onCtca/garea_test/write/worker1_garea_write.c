#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "ctca.h"
#include "../garea.h"

int main()
{
    int myrank, nprocs, fromrank, gareaid0, gareaid1, gareaid2;
    int intparams[2];
    int garea_params[1];
    int *data;
    int c, datasize;
    int garea_params_num = 1;
    size_t data_element = 5;

    data = (int *)malloc(data_element * sizeof(int));

    // progid=0,サブコミュニケータ当たりのプロセス数=4
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

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

    MPI_Finalize();
}
