#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

float output_float_time(struct timespec start_time, struct timespec end_time)
{
    long seconds = end_time.tv_sec - start_time.tv_sec;
    long nanoseconds = end_time.tv_nsec - start_time.tv_nsec;
    if (nanoseconds < 0)
    {
        seconds--;
        nanoseconds += 1000000000L; // 1秒を追加
    }
    return seconds + nanoseconds * 1e-9;
}

void calc_measure_time(float *elapsed_time, int iteration, float *average_time, float *max_time, float *min_time)
{
    float sum_time = 0.0;
    *max_time = elapsed_time[0];
    *min_time = elapsed_time[0];

    for (int i = 0; i < iteration; i++)
    {
        // 最大値の更新
        if (elapsed_time[i] > *max_time)
        {
            *max_time = elapsed_time[i];
        }

        // 最小値の更新
        if (elapsed_time[i] < *min_time)
        {
            *min_time = elapsed_time[i];
        }
        sum_time += elapsed_time[i];
    }
    *average_time = sum_time / (float)iteration;
}

// ローカル時間の中で最も大きい値を持つプロセスの値を返す
float reduce_max_gtime(float *ltime_list, float *max_gtime, int step_num, MPI_Comm new_comm)
{
    for (int i = 0; i < step_num; i++)
    {
        MPI_Reduce(&ltime_list[i], &max_gtime[i], 1, MPI_FLOAT, MPI_MAX, 0, new_comm);
    }
}