#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>

#include "ctca.h"
#include "buffer.h"

int main(int argc, char *argv[])
{
    int subnprocs, submyrank;

    bool data_direction = true;
    int requester_data_division[3];
    requester_data_division[0] = 1;
    requester_data_division[1] = 1;
    requester_data_division[2] = 1;
    int *local_data_H, *local_data_D;
    CTCAR_init();
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(CTCA_subcomm, &submyrank);

    int step = 0;
    size_t global_data_element = 2 * 2 * 2;
    size_t global_data_size = global_data_element * sizeof(int);
    for (int i = 0; i < 2; i++)
    {

        printf("req%d: step%d do\n", submyrank, i);
        local_data_H = (int *)malloc(global_data_size);
        cudaMalloc((void **)&local_data_D, global_data_size);

        for (int j = 0; j < global_data_element; j++)
        {
            local_data_H[j] = 1;
        }
        cudaMemcpy(local_data_D, local_data_H, global_data_size, cudaMemcpyHostToDevice);

        requester_buffer_init_withint(requester_data_division, global_data_element, data_direction);
        requester_buffer_write(local_data_D, step);
        MPI_Barrier(MPI_COMM_WORLD);
        requester_buffer_fin();
        free(local_data_H);
        cudaFree(local_data_D);
        CTCAR_finalize();
    }

    printf("req%d:finalize done\n", submyrank);
}
