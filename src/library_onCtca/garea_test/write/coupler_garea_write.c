#include <stdio.h>
#include <mpi.h>
#include "ctca.h"
#include "../garea.h"

int main()
{
    int myrank, nprocs, progid, fromrank, gareaid0, gareaid1, gareaid2;
    int intparams[2];
    int reqinfo[CTCAC_REQINFOITEMS]; // reqinf[4]
    int data[6 * 400];
    int intparams_num = 2;
    int garea_params[1];
    int garea_params_num = 1;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    printf("cpr%d: garea_create do\n", myrank);
    CTCAC_garea_create(&gareaid0);
    printf("cpr%d: garea_create done\n", myrank);

    // garea_write debug
    MPI_Barrier(MPI_COMM_WORLD);

    CTCAC_garea_delete();

    printf("cpr%d: finalize\n", myrank);
    MPI_Finalize();
}
