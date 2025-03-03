#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include "ctca.h"
#include "buffer.h"

int main(int argc, char *argv[])
{
    int subnprocs, submyrank;
    int worker_data_division[3];
    target_side_type target_side;
    int progid = 0;
    int x_division = atoi(argv[1]);
    int y_division = 1;
    int z_division = atoi(argv[2]);
    int procs_per_subcomm = atoi(argv[3]);
    int *dest_data;
    int num;
    int length_x;
    int length_y;
    int length_z;
    char step_name[200];
    char file_name[200];
    int fromrank;
    int intparams[2];
    int intparamnum = 2;
    int max_element = 4096;
    int skip = 100;
    int loop = 1000;
    CTCAW_init(progid, procs_per_subcomm);
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(CTCA_subcomm, &submyrank);

    if (procs_per_subcomm < subnprocs)
    {
        printf("numの要素数がworkerのプロセス数より小さいです(procs_per_subcomm=%d, wrk_subnprocs=%d)\n", procs_per_subcomm, subnprocs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    worker_data_division[0] = x_division;
    worker_data_division[1] = y_division;
    worker_data_division[2] = z_division;
    target_side = (target_side_type)ZX;

    int step = 0;
    worker_buffer_init_withint(worker_data_division, target_side, &num);

    size_t global_data_element = num * num * num;
    size_t local_data_element = num * num * num / subnprocs;
    size_t local_data_byte = local_data_element * sizeof(int);
    dest_data = (int *)malloc(local_data_byte);
    for (int i = 0; i < local_data_element; i++)
    {
        dest_data[i] = -1;
    }
    for (int step = 0; step < loop + skip; step++)
    {
        worker_buffer_read(dest_data, step);
    }
    worker_print_detail_read_time();
    MPI_Barrier(MPI_COMM_WORLD);
    worker_buffer_fin();
    free(dest_data);

    CTCAW_finalize();
}
