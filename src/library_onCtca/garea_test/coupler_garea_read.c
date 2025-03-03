#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include "ctca.h"
#include "garea.h"

int main()
{
    int myrank, nprocs;
    int *gareaid;
    int num_gareas = 2;

    gareaid = (int* )malloc(num_gareas*sizeof(int));


    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    printf("coupler: myrank=%d\n",myrank);
    CTCAC_garea_create(&gareaid[0]);
    CTCAC_garea_create(&gareaid[1]);
    printf("coupler: create done\n");

    //garea_read debug
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    CTCAC_garea_delete();

    fprintf(stderr, "%d: coupler finalize\n", myrank);
    MPI_Finalize();
}
