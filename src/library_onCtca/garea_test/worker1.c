#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include "ctca.h"
#include "garea.h"

int main()
{
    int myrank, nprocs, fromrank, gareaid,i;
    int intparams[1];
    int garea_intparams[1];
    int* data;
    int c, datasize;
    int garea_intparams_num=1;
    size_t data_element;

    data = (int* )malloc(data_element*sizeof(int));

    //progid=0,サブコミュニケータ当たりのプロセス数=4
    CTCAW_init(0,1);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    printf("worker1: myrank=%d\n",myrank);
    CTCAW_garea_create(&gareaid);
    printf("worker1: create done\n");

    int target_world_rank=0; 
    size_t offset=0;
    
 
    printf("worker1: pollreq do\n");
    CTCAW_pollreq(&fromrank, garea_intparams, garea_intparams_num);
    printf("worker1: pollreq done\n");

    data_element = garea_intparams[0];

    for(int i=0;i<data_element;i++){
        data[i]=1000+i;
    }

    printf("worker1: complete do\n");
    CTCAW_complete();
    printf("worker1: complete\n");

    printf("worker1: garea_attach do\n");
    CTCAW_garea_attach(gareaid,data,data_element*sizeof(int));
    printf("worker1: garea_attach done\n");

    MPI_Barrier(MPI_COMM_WORLD);

    for(i=0;i<data_element;i++){
        printf("worker1: data[%d]=%d\n",i,data[i]);
    }
 
    CTCAR_garea_detach(gareaid, data);
    printf("CTCAR_garea_detach done\n");

    CTCAW_garea_delete();

    while(1){
        CTCAW_pollreq(&fromrank, garea_intparams, garea_intparams_num);

        if (CTCAW_isfin()) 
            break;
    }
    
    fprintf(stderr, "%d: worker1 finalize\n", myrank);
    CTCAW_finalize();
}
