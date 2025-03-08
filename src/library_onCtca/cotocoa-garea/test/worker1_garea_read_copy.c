#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "ctca.h"

int main()
{
    int world_myrank, world_nprocs, fromrank;
    int local_myrank,local_nprocs;
    int *data1, *data2, *gareaid;
    size_t data_element=10;
    int num_gareas = 2;
    size_t offset=0;
    int target_world_rank;
    int procspercomm = 2;
    
    data1 = (int* )malloc(data_element*sizeof(int));
    data2 = (int* )malloc(data_element*sizeof(int));
    gareaid = (int* )malloc(num_gareas*sizeof(int));

    //progid=0,サブコミュニケータ当たりのプロセス数=4
    CTCAW_init(0, procspercomm);
    MPI_Comm_size(MPI_COMM_WORLD,&world_nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_myrank);
    MPI_Comm_size(CTCA_subcomm, &local_nprocs);
    MPI_Comm_rank(CTCA_subcomm,&local_myrank);

    printf("worker1: world_myrank=%d\n",world_myrank);
    CTCAW_garea_create(&gareaid[0]);
    CTCAW_garea_create(&gareaid[1]);
    printf("worker1: create done\n");


    //garea_read debug
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i=0;i<data_element;i++){
        data1[i]=1000*(world_myrank+1)+i;
        data2[i]=2000*(world_myrank+1)+i;
    }

    CTCAW_get_grank_req(0, &target_world_rank);
    
    if(local_myrank==0){
        printf("worker%d gareaid%d: garea_attach do\n",world_myrank,gareaid[0]);
        CTCAW_garea_attach(gareaid[0],(void* )data2,data_element*sizeof(int));
        //CTCAW_garea_attach(gareaid[1],(void* )data2,data_element*sizeof(int));
        printf("worker%d gareaid%d: garea_attach done\n",world_myrank,gareaid[0]);
    }

    // printf("worker gareaid%d: garea_read do\n",gareaid[0]);
    // CTCAW_garea_read_int(gareaid[0], target_world_rank, offset, data_element, data1);
    // printf("worker gareaid%d: garea_read done\n",gareaid[0]);
    CTCAW_garea_read_int(gareaid[1],0,0,data_element,data1);
    MPI_Barrier(MPI_COMM_WORLD);
    if(local_myrank == 0){
        show_base_addresses();
        for(int i=0;i<data_element;i++){
            printf("wrk%d: data2[%d]=%d\n",local_myrank,i,data2[i]);
        }
        MPI_Barrier(CTCA_subcomm);
    }else if(local_myrank==1){
        MPI_Barrier(CTCA_subcomm);
        for(int i=0;i<data_element;i++){
            printf("wrk%d: data1[%d]=%d\n",local_myrank,i,data1[i]);
        }
    }

    // for(int i=0;i<data_element;i++){
    //     printf("worker_read: data1[%d]=%d\n",i,data1[i]);
    // }

    // printf("%d worker detach do\n",world_myrank);
    // CTCAR_garea_detach(gareaid[0], data2);
    // printf("%d worker detach done\n",world_myrank);

    CTCAW_garea_delete();
    fprintf(stderr, "%d: worker1 finalize\n", world_myrank);
    CTCAW_finalize();
}
