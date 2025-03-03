#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "ctca.h"

#define DEF_MAXNUMAREAS 10
#define DEF_MAXINTPARAMS 10
#define DEF_CPL_DATBUF_SLOTSZ 80000

int main(int argc, char *argv[])
{
    int myrank, nprocs, areaid, progid, fromrank;
    int intparams[2];
    int reqinfo[CTCAC_REQINFOITEMS]; // reqinf[4]
    const int num = atoi(argv[1]);
    const int step_num = atoi(argv[2]);
    const int whole_surface_num = num * num;
    float *result;
    int numintparams = 2;

    result = (float *)malloc(sizeof(float) * whole_surface_num);
    CTCAC_init_detail(DEF_MAXNUMAREAS, 20000, DEF_MAXINTPARAMS, DEF_CPL_DATBUF_SLOTSZ, 20000);

    MPI_Comm_size(CTCA_subcomm, &nprocs);
    MPI_Comm_rank(CTCA_subcomm, &myrank);

    CTCAC_regarea_real4(&areaid);
    MPI_Barrier(MPI_COMM_WORLD);
    while (1)
    {
        CTCAC_pollreq(reqinfo, &fromrank, intparams, numintparams);

        if (CTCAC_isfin())
            break;

        if (fromrank >= 0)
        {
            progid = intparams[0];
            CTCAC_enqreq(reqinfo, progid, intparams, numintparams);
        }
    }
    CTCAC_finalize();
}
