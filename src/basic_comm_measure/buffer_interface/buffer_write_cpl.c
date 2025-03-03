#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <mpi.h>
#include "ctca.h"
#include "buffer.h"

int main(int argc, char *argv[])
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

    coupler_buffer_init();

    MPI_Barrier(MPI_COMM_WORLD);
    coupler_buffer_fin();
    step++;

    CTCAC_finalize();
}