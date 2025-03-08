#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "ctca.h"

int main()
{
    int myrank, nprocs, fromrank;
    int *data1, *data2, *gareaid;
    size_t data_element=10;
    int num_gareas = 2;
    size_t offset=0;
    int target_world_rank;
    
    data1 = (int* )malloc(data_element*sizeof(int));
    data2 = (int* )malloc(data_element*sizeof(int));
    gareaid = (int* )malloc(num_gareas*sizeof(int));

    //progid=0,サブコミュニケータ当たりのプロセス数=4
    CTCAW_init(0, 4);
    MPI_Comm_size(CTCA_subcomm,&nprocs);
    MPI_Comm_rank(CTCA_subcomm,&myrank);

    printf("worker1: myrank=%d\n",myrank);
    CTCAW_garea_create(&gareaid[0]);
    CTCAC_garea_create(&gareaid[1]);
    printf("worker1: create done\n");


    //garea_read debug
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i=0;i<data_element;i++){
        data1[i]=1000*(myrank+1)+i;
        data2[i]=2000*(myrank+1)+i;
    }

    CTCAW_get_grank_req(0, &target_world_rank);
    
    printf("worker gareaid%d: garea_attach do\n",gareaid[1]);
    CTCAW_garea_attach(gareaid[1],data2,data_element*sizeof(int));
    printf("worker gareaid%d: garea_attach done\n",gareaid[1]);

    printf("worker gareaid%d: garea_read do\n",gareaid[0]);
    CTCAW_garea_read_int(gareaid[0], target_world_rank, offset, data_element, data1);
    printf("worker gareaid%d: garea_read done\n",gareaid[0]);

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0;i<data_element;i++){
        printf("worker_read: data1[%d]=%d\n",i,data1[i]);
    }


    CTCAR_garea_detach(gareaid[1], data2);

    CTCAW_garea_delete();
    fprintf(stderr, "%d: worker1 finalize\n", myrank);
    CTCAW_finalize();
}
