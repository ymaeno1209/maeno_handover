#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include "ctca.h"
#include "buffer.h"

int main(int argc, char *argv[])
{

    int progid = 0;
    int procs_per_subcomm = 1;
    int subnprocs, submyrank;
    int worker_data_division[3];
    int num;
    target_side_type target_side;

    worker_data_division[0] = 1;
    worker_data_division[1] = 1;
    worker_data_division[2] = 1;
    target_side = (target_side_type)ZX;

    CTCAW_init(progid, procs_per_subcomm);
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(CTCA_subcomm, &submyrank);

    int step = 0;
    for (int i = 0; i < 2; i++)
    {

        printf("wrk%d: step%d do\n", submyrank, i);
        int *dest_data = (int *)malloc(sizeof(int) * 100);
        worker_buffer_init_withint(worker_data_division, target_side, &num);
        worker_buffer_read(dest_data, step);
        MPI_Barrier(MPI_COMM_WORLD);
        worker_buffer_fin();
        free(dest_data);
        CTCAW_finalize();
    }

    printf("wrk%d: finalize done\n", submyrank);
}
