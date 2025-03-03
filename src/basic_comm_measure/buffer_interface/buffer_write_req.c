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
    int *data_Host;
    int *data_Device;
    int num = atoi(argv[1]);
    size_t global_data_element = num * num * num;

    bool data_direction;
    int requester_data_division[3];
    struct timeval tv0_write;
    struct timeval tv1_write;
    int *local_data, *remote_data;
    int *local_data_H, *remote_data_H;
    int skip = 100;
    int loop = 1000;
    int max_element = 4096;
    double start_time, end_time;
    double elapsed_time = 0.0;
    CTCAR_init();

    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(CTCA_subcomm, &submyrank);
    if (num < subnprocs)
    {
        printf("numの要素数がrequesterのプロセス数より小さいです(num=%d, req_subnprocs=%d)\n", num, subnprocs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int x_division = 1;
    int y_division = 1;
    int z_division = subnprocs;
    data_direction = true;
    requester_data_division[0] = x_division;
    requester_data_division[1] = y_division;
    requester_data_division[2] = z_division;
    size_t local_data_element = global_data_element / z_division;

    size_t data_byte = global_data_element * sizeof(int);
    local_data_H = (int *)malloc(data_byte);
    cudaMalloc((void **)&local_data, data_byte);
    for (int i = 0; i < global_data_element; i++)
    {
        local_data_H[i] = 0;
    }
    cudaMemcpy(local_data, local_data_H, data_byte, cudaMemcpyHostToDevice);
    requester_buffer_init_withint(requester_data_division, global_data_element, data_direction);

    for (int step = 0; step < loop + skip; step++)
    {
        if (step == skip)
        {
            MPI_Barrier(CTCA_subcomm);
            start_time = MPI_Wtime();
        }
        requester_buffer_write(local_data, step);
    }
    MPI_Barrier(CTCA_subcomm);
    end_time = MPI_Wtime();
    elapsed_time = ((end_time - start_time) / loop) * 1000 * 1000;
    if (submyrank == 0)
    {
        printf("num=%d, data_byte=%d[byte], time=%f[μs]\n", num, data_byte, elapsed_time);
    }
    requester_print_detail_write_time(skip, loop);
    count_buffer_index();

    MPI_Barrier(MPI_COMM_WORLD);
    requester_buffer_fin();
    free(local_data_H);
    cudaFree(local_data);

    CTCAR_finalize();

    cudaFree(data_Device);
}
