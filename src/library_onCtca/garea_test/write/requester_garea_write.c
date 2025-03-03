#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include "ctca.h"
#include "../garea.h"

int main()
{
    int myrank, nprocs, progid, gareaid0, gareaid1, gareaid2;
    int intparams[2];
    int garea_params[1];
    size_t data_element = 5;
    int *data;
    int i, j, k;
    int prognum = 1;
    int intparams_num = 2;
    int target_world_rank;

    data = (int *)malloc(data_element * sizeof(int));

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    target_world_rank = 5 + myrank;

    printf("req%d: garea_create do\n", myrank);
    CTCAR_garea_create(&gareaid0);
    printf("req%d: garea_create done\n", myrank);

    for (i = 0; i < data_element; i++)
        data[i] = i;

    printf("req%d: garea_attach do\n", myrank);
    CTCAR_garea_attach(gareaid0, data, data_element * sizeof(int));
    printf("req%d: garea_attach done\n", myrank);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < data_element; i++)
    {
        printf("req%d: data[%d]=%d\n", myrank, i, data[i]);
    }

    CTCAR_garea_detach(gareaid0, data);
    printf("req%d garea_detach done\n", myrank);

    CTCAR_garea_delete();
    printf("req%d: garea_delete done\n", myrank);

    MPI_Finalize();
}
