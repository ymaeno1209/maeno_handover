#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "ctca.h"
#include "garea.h"

int main()
{
    int myrank, nprocs, progid,gareaid,wgid;
    int garea_intparams[1];
    int garea_intparams_num=1;
    size_t data_element=10;
    int* data;
    int i, j, k;
    int prognum = 1;
    int intparams_num=2;
    int target_world_rank=2; 
    size_t offset=0;


    data = (int* )malloc(data_element*sizeof(int));
    for(i=0;i<data_element;i++) data[i] = i;
    garea_intparams[0] = data_element;

    CTCAR_init();
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    printf("requester: myrank=%d\n",myrank);

    CTCAR_garea_create(&gareaid);
    printf("requeseter: create done\n");

    if(myrank == 0){
        printf("requester: sendreq do\n");
        CTCAR_sendreq(garea_intparams,garea_intparams_num);
        printf("requester: sendreq done\n");

        //to finish coupler
        printf("requester: sendreq do\n");
        CTCAR_sendreq(garea_intparams,garea_intparams_num);
        printf("requester: sendreq done\n");

        wgid = 0;
    }
    printf("requester: garea_write do\n");
    CTCAR_garea_write_int(gareaid, target_world_rank, offset, data_element, data);
    printf("requester: garea_write done\n");

    MPI_Barrier(MPI_COMM_WORLD);

    CTCAR_garea_delete();
    fprintf(stderr, "%d: requester finalize\n", myrank);

    CTCAR_finalize();
}
