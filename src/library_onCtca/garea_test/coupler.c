#include <stdio.h>
#include <mpi.h>
#include "ctca.h"
#include "garea.h"

int main()
{
    int myrank, nprocs, fromrank, gareaid;
    int reqinfo[CTCAC_REQINFOITEMS];//reqinfo[4]
    int data[6*400];
    int intparams_num=2;
    int garea_intparams[1];
    int garea_intparams_num = 1;
    int progid = 0;

    CTCAC_init();
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    printf("coupler: myrank=%d\n",myrank);
    CTCAC_garea_create(&gareaid);
    printf("coupler: create done\n");

    printf("coupler pollreq do\n");
    CTCAC_pollreq(reqinfo, &fromrank, garea_intparams, garea_intparams_num);//requesterからデータを受け取る
    printf("coupler pollreq done\n");

    printf("coupler enqreq do\n");
    CTCAC_enqreq(reqinfo, progid, garea_intparams, garea_intparams_num);//enq
    printf("coupler enqreq done\n");

    printf("coupler pollreq do\n");
    CTCAC_pollreq(reqinfo, &fromrank, garea_intparams, garea_intparams_num);//workerにデータを送る
    printf("coupler pollreq done\n");

    MPI_Barrier(MPI_COMM_WORLD);

    CTCAC_garea_delete();

    while(1){
        CTCAC_pollreq(reqinfo, &fromrank, garea_intparams, garea_intparams_num);

        if (CTCAC_isfin()) 
            break;
    }

    fprintf(stderr, "%d: coupler finalize\n", myrank);
    CTCAC_finalize();
}
