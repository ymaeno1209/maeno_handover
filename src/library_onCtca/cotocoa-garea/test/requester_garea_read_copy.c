#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "ctca.h"

int main()
{
    int world_myrank, world_nprocs, progid, i, target_world_rank;
    int local_myrank,local_nprocs;
    int *gareaid, *data1, *data2;
    size_t data_element=10;
    size_t offset = 0;
    size_t data_size = data_element*sizeof(int);
    int num_gareas = 2;
    int worker_localrank = 0;
    int wgid = 0;

    data1 = (int* )malloc(data_size);
    data2 = (int* )malloc(data_size);
    gareaid = (int* )malloc(num_gareas*sizeof(int));

    CTCAR_init();

    MPI_Comm_size(MPI_COMM_WORLD,&world_nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_myrank);
    MPI_Comm_size(CTCA_subcomm, &local_nprocs);
    MPI_Comm_rank(CTCA_subcomm, &local_myrank);

    printf("requester: world_myrank=%d\n",world_myrank);
    CTCAR_garea_create(&gareaid[0]);
    CTCAR_garea_create(&gareaid[1]);
    printf("requeseter: create done\n");

    for(i=0;i<data_element;i++) data1[i] = i;
    for(i=0;i<data_element;i++) data2[i] = i+500;
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("requester: garea_attach do\n");
    CTCAR_garea_attach(gareaid[1],data1,data_size);
    printf("requester: garea_attach done\n");

    printf("requester gareaid%d: garea_read do\n",gareaid[1]);
    CTCAR_get_grank_wrk(wgid, worker_localrank, &target_world_rank);
    CTCAR_garea_write_int(gareaid[0], target_world_rank, offset, data_element, data2);
    printf("requester gareaid%d: garea_read done\n",gareaid[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    // for(i=0;i<data_element;i++){
    //     printf("requester_read: data2[%d]=%d\n",i,data2[i]);
    // }

    // CTCAR_garea_detach(gareaid[0], data1);
    // printf("CTCAR_garea_detach done\n");

    CTCAR_garea_delete();
    fprintf(stderr, "%d: requester finalize\n", world_myrank);

    CTCAR_finalize();
}
