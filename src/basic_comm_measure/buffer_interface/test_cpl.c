#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <mpi.h>
#include "ctca.h"
#include "buffer.h"

int main()
{

    int subnprocs, submyrank, progid, fromrank;
    int reqinfo[CTCAC_REQINFOITEMS];
    int intparams[2];
    int numintparams = 2;
    int max_element = 4096;
    CTCAC_init();
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(CTCA_subcomm, &submyrank);
    int step = 0;
    for (int i = 0; i < 2; i++)
    {

        printf("cpl%d: buffer_init do on step%d\n", submyrank, step);
        coupler_buffer_init();
        printf("cpl%d: buffer_init done on step%d\n", submyrank, step);

        MPI_Barrier(MPI_COMM_WORLD);
        coupler_buffer_fin();
        step++;
        CTCAC_finalize();
    }
    fprintf(stderr, "cpl%d: finalize done\n", submyrank);
}