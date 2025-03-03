#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "ctca.h"
#include "garea.h"

int main()
{
    int myrank, nprocs, progid, i, target_world_rank;
    int *gareaid, *data1, *data2;
    size_t data_element=10;
    size_t offset;
    int num_gareas = 2;

    data1 = (int* )malloc(data_element*sizeof(int));
    data2 = (int* )malloc(data_element*sizeof(int));
    gareaid = (int* )malloc(num_gareas*sizeof(int));

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    printf("requester: myrank=%d\n",myrank);
    CTCAR_garea_create(&gareaid[0]);
    CTCAR_garea_create(&gareaid[1]);
    printf("requeseter: create done\n");

    for(i=0;i<data_element;i++) data1[i] = i;
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("requester: garea_attach do\n");
    CTCAR_garea_attach(gareaid[0],data1,data_element*sizeof(int));
    printf("requester: garea_attach done\n");

    printf("requester gareaid%d: garea_read do\n",gareaid[1]);
    target_world_rank = 2;
    offset = 0;
    CTCAR_garea_read_int(gareaid[1], target_world_rank, offset, data_element, data2);
    printf("requester gareaid%d: garea_read done\n",gareaid[1]);

    MPI_Barrier(MPI_COMM_WORLD);

    for(i=0;i<data_element;i++){
        printf("requester_read: data2[%d]=%d\n",i,data2[i]);
    }

    CTCAR_garea_detach(gareaid[0], data1);
    printf("CTCAR_garea_detach done\n");

    CTCAR_garea_delete();
    fprintf(stderr, "%d: requester finalize\n", myrank);

    MPI_Finalize();
}
