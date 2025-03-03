#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <pthread.h>
#include <inttypes.h>
#include <limits.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include "sim_diffusion.h"
#include "sim_misc.h"
#include "kernel.h"
#include "ctca.h"
#include "util.h"

#define CON_PACKING_COUNT_MAX 128
#define CTCA_REGAREA_MAX_STEP 3
void write_to_file(const float *data, int num, int lenx, int lenz, int rank, int y, int step);
void initialize_con_packing_count(int *con_packing_count, int num);

int main(int argc, char *argv[])
{
    const int num = atoi(argv[1]);
    const int step_num = atoi(argv[2]);
    const int sim_times_per_step = atoi(argv[3]);
    const int progid = 0;
    const int numintparams = 2;
    int intparams[2];
    intparams[0] = progid;
    int areaid;
    struct timespec req_main_start_time, req_main_end_time;
    struct timespec coupled_app_start_time, coupled_app_end_time;
    struct timespec req_ctca_init_start_time, req_ctca_init_end_time;
    struct timespec req_buf_write_start_time, req_buf_write_end_time;
    struct timespec req_buf_calc_start_time, req_buf_calc_end_time;
    struct timespec req_packing_start_time, req_packing_end_time;
    struct timespec req_reshape_start_time, req_reshape_end_time;
    struct timespec req_polling_start_time, req_polling_end_time;
    struct timespec req_enq_start_time, req_enq_end_time;

    float *req_buf_write_time_list;
    float *req_buf_calc_time_list;
    float *req_packing_time_list;
    float *req_reshape_time_list;
    float *req_polling_time_list;
    float *req_enq_time_list;
    req_buf_write_time_list = (float *)malloc(step_num * sizeof(float));
    req_buf_calc_time_list = (float *)malloc(step_num * sizeof(float));
    req_packing_time_list = (float *)malloc(step_num * sizeof(float));
    req_reshape_time_list = (float *)malloc(step_num * sizeof(float));
    req_polling_time_list = (float *)malloc(step_num * sizeof(float));
    req_enq_time_list = (float *)malloc(step_num * sizeof(float));
    for (int i = 0; i < step_num; i++)
    {
        req_packing_time_list[i] = 0.0;
        req_reshape_time_list[i] = 0.0;
        req_polling_time_list[i] = 0.0;
        req_enq_time_list[i] = 0.0;
    }

    // cudaに関する初期設定
    static cudaStream_t top_stream;
    static cudaStream_t mid_stream;
    static cudaStream_t bot_stream;
    static cudaStream_t packing_stream;
    CUdevice cuDevice;
    CUcontext cuContext;
    int local_rank = -1;
    int ngpus = 0;
    int gpuid = 0;
    int i, j, k, r, y;
    int new_nprocs, new_rank;
    local_rank = omb_get_local_rank();
    if (local_rank >= 0)
    {
        cudaGetDeviceCount(&ngpus);
        gpuid = local_rank % ngpus;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_ctca_init_start_time);
    CTCAR_init_detail(10, 4094, 10);
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_ctca_init_end_time);

    cuInit(0);
    cudaSetDevice(gpuid);

    cudaStreamCreate(&top_stream);
    cudaStreamCreate(&mid_stream);
    cudaStreamCreate(&bot_stream);
    cudaStreamCreate(&packing_stream);
    MPI_Comm_size(CTCA_subcomm, &new_nprocs);
    MPI_Comm_rank(CTCA_subcomm, &new_rank);
    char hostname[128];
    gethostname(hostname, sizeof(hostname));
    fprintf(stdout, "Sim : new_rank %d: hostname = %s, gpuid = %d\n", new_rank, hostname, gpuid);
    fflush(stdout);

    const int rank_up = new_rank != new_nprocs - 1 ? new_rank + 1 : MPI_PROC_NULL;
    const int rank_down = new_rank != 0 ? new_rank - 1 : MPI_PROC_NULL;
    const int nx0 = num;
    const int ny0 = nx0;
    const int nz0 = nx0;
    const int whole_surface_num = num * num;
    const size_t whole_num = (size_t)num * (size_t)num * (size_t)num;

    const int nx = nx0;
    const int ny = ny0;
    const int nz = nz0 / new_nprocs;
    const int procs_surface_num = nx * nz;

    const int mgn = 1;
    const int lnx = nx;
    const int lny = ny;
    const int lnz = nz + 2 * mgn;
    const int ln = lnx * lny * lnz;

    const float lx = 1.0;
    const float ly = 1.0;
    const float lz = 1.0;

    const float dx = lx / (float)nx0;
    const float dy = ly / (float)ny0;
    const float dz = lz / (float)nz0;

    const float kappa = 0.1;
    const float dt = 0.1 * fmin(fmin(dx * dx, dy * dy), dz * dz) / kappa;

    double time = 0.0;
    double flop = 0.0;
    double elapsed_time = 0.0;

    const int tag = 0;
    MPI_Status stat[4];
    int step = 0;
    float *f, *fn, *f_H, *y_axis_normal_plane_perprocs, *y_axis_normal_plane_allprocs, *y_axis_normal_plane_allprocs_H, *data_allprocs_allstep;
    float *allprocs_commdata_tmp;
    size_t data_allprocs_allstep_size = sizeof(float) * whole_num * (size_t)CTCA_REGAREA_MAX_STEP;
    cudaMalloc((void **)&f, sizeof(float) * ln);
    cudaMalloc((void **)&fn, sizeof(float) * ln);
    cudaMalloc((void **)&y_axis_normal_plane_perprocs, sizeof(float) * procs_surface_num);
    cudaMalloc((void **)&y_axis_normal_plane_allprocs, sizeof(float) * whole_surface_num);
    cudaMalloc((void **)&data_allprocs_allstep, data_allprocs_allstep_size);
    cudaMalloc((void **)&allprocs_commdata_tmp, (size_t)sizeof(float) * (size_t)num * (size_t)num * (size_t)num);
    f_H = (float *)malloc(sizeof(float) * ln);
    y_axis_normal_plane_allprocs_H = (float *)malloc(sizeof(float) * whole_surface_num);
    float *reverse_slice_f = (float *)malloc(sizeof(float) * nx * nz);
    float *result = (float *)malloc(sizeof(float) * nx0 * nz0);
    int collect_data_count = (new_rank == 0) ? (new_nprocs - 1) * nz : nz;
    float *f_debug = (float *)malloc(num * num * num * sizeof(float));
    int con_packing_count;
    initialize_con_packing_count(&con_packing_count, num);
    fprintf(stderr, "num=%d,con_packing_count=%d\n", num, con_packing_count);

    MPI_Request *collect_req = (MPI_Request *)malloc(collect_data_count * sizeof(MPI_Request));
    for (size_t i = 0; i < collect_data_count; i++)
    {
        collect_req[i] = MPI_REQUEST_NULL;
    }
    MPI_Status *collect_stat = (MPI_Status *)malloc(collect_data_count * sizeof(MPI_Status));
    size_t *ctca_regarea_hdl;
    ctca_regarea_hdl = (size_t *)malloc(sizeof(size_t) * ny * CTCA_REGAREA_MAX_STEP);
    // GPUメモリに直接初期化

    call_kernel_init(new_rank, nx, ny, nz, mgn, dx, dy, dz, f);
    CTCAR_regarea_real4(data_allprocs_allstep, whole_num * (size_t)CTCA_REGAREA_MAX_STEP, &areaid);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC_RAW, &coupled_app_start_time);
    MPI_Barrier(CTCA_subcomm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_main_start_time);
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_write_start_time);
    int hdl_num = 0;
    int hdl_slot = 0;
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_packing_start_time);
    MPI_Gather(&f[nx * ny], nx * ny * nz, MPI_FLOAT, allprocs_commdata_tmp, nx * ny * nz, MPI_FLOAT, 0, CTCA_subcomm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_packing_end_time);
    req_packing_time_list[step] += output_float_time(req_packing_start_time, req_packing_end_time);
    for (y = 0; y < ny; y++)
    {
        intparams[1] = hdl_slot * ny + y;
        if (hdl_slot == CTCA_REGAREA_MAX_STEP)
        {
            hdl_num = 0;
            hdl_slot = 0;
        }
        // fからy軸に垂直な一面を取り出す

        // 実装予定
        // 各プロセスの変数fに格納されているデータをプロセス番号0に集める

        clock_gettime(CLOCK_MONOTONIC_RAW, &req_reshape_start_time);
        if (new_rank == 0)
        {
            call_con_packing_data(&data_allprocs_allstep[hdl_slot * whole_num], allprocs_commdata_tmp, num, packing_stream);
            cudaStreamSynchronize(packing_stream);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &req_reshape_end_time);
        req_reshape_time_list[step] += output_float_time(req_reshape_start_time, req_reshape_end_time);
        // couplerへデータをrequest
        if (new_rank == 0)
        {
            // printf("step=%d, y=%d, hdl_num=%d, hdl_slot=%d\n", step, y, hdl_num, hdl_slot);
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_polling_start_time);
            while (1)
            {
                if (CTCAR_test(ctca_regarea_hdl[hdl_num]))
                    break;
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_polling_end_time);
            req_polling_time_list[step] += output_float_time(req_polling_start_time, req_polling_end_time);

            clock_gettime(CLOCK_MONOTONIC_RAW, &req_enq_start_time);
            CTCAR_sendreq_hdl(intparams, numintparams, &ctca_regarea_hdl[hdl_num]);
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_enq_end_time);
            req_enq_time_list[step] += output_float_time(req_enq_start_time, req_enq_end_time);
            hdl_num++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_write_end_time);
    req_buf_write_time_list[step] = output_float_time(req_buf_write_start_time, req_buf_write_end_time);

    clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_calc_start_time);

    int icnt = 1;
    while (1)
    {
        MPI_Status stat[4];
        MPI_Request req[4];

        cudaStreamSynchronize(top_stream);
        cudaStreamSynchronize(bot_stream);

        // Halo communication
        MPI_Irecv(&f[0], nx * ny, MPI_FLOAT, rank_down, tag, CTCA_subcomm, &req[0]);
        MPI_Irecv(&f[nx * ny * (nz + mgn)], nx * ny, MPI_FLOAT, rank_up, tag, CTCA_subcomm, &req[1]);
        MPI_Isend(&f[nx * ny * nz], nx * ny, MPI_FLOAT, rank_up, tag, CTCA_subcomm, &req[2]);
        MPI_Isend(&f[nx * ny * mgn], nx * ny, MPI_FLOAT, rank_down, tag, CTCA_subcomm, &req[3]);
        MPI_Waitall(4, req, stat);

        cudaStreamSynchronize(mid_stream);

        if (icnt % sim_times_per_step == 1 && icnt != 1)
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_calc_end_time);
            req_buf_calc_time_list[step] = output_float_time(req_buf_calc_start_time, req_buf_calc_end_time);

            step++;
            hdl_slot++;
            // fをy軸に垂直な面で切り、面ごとにcouplerへ送る
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_write_start_time);
            if (hdl_slot == CTCA_REGAREA_MAX_STEP)
            {
                hdl_num = 0;
                hdl_slot = 0;
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_packing_start_time);
            MPI_Gather(&f[nx * ny], nx * ny * nz, MPI_FLOAT, allprocs_commdata_tmp, nx * ny * nz, MPI_FLOAT, 0, CTCA_subcomm);
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_packing_end_time);
            req_packing_time_list[step] += output_float_time(req_packing_start_time, req_packing_end_time);
            for (y = 0; y < ny; y++)
            {
                intparams[1] = hdl_slot * ny + y;
                // データの整形

                clock_gettime(CLOCK_MONOTONIC_RAW, &req_reshape_start_time);
                if (new_rank == 0)
                {
                    call_con_packing_data(&data_allprocs_allstep[hdl_slot * whole_num], allprocs_commdata_tmp, num, packing_stream);
                    cudaStreamSynchronize(packing_stream);
                }
                clock_gettime(CLOCK_MONOTONIC_RAW, &req_reshape_end_time);
                req_reshape_time_list[step] += output_float_time(req_reshape_start_time, req_reshape_end_time);
                // couplerへデータをrequest
                if (new_rank == 0)
                {
                    // printf("step=%d, y=%d,hdl_num=%d, hdl_slot=%d\n", step, y, hdl_num, hdl_slot);
                    clock_gettime(CLOCK_MONOTONIC_RAW, &req_polling_start_time);
                    while (1)
                    {
                        if (CTCAR_test(ctca_regarea_hdl[hdl_num]))
                            break;
                    }
                    clock_gettime(CLOCK_MONOTONIC_RAW, &req_polling_end_time);
                    req_polling_time_list[step] += output_float_time(req_polling_start_time, req_polling_end_time);
                    clock_gettime(CLOCK_MONOTONIC_RAW, &req_enq_start_time);
                    CTCAR_sendreq_hdl(intparams, numintparams, &ctca_regarea_hdl[hdl_num]);
                    clock_gettime(CLOCK_MONOTONIC_RAW, &req_enq_end_time);
                    hdl_num++;
                }
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_write_end_time);
            req_buf_write_time_list[step] = output_float_time(req_buf_write_start_time, req_buf_write_end_time);
            if (step == step_num - 1)
                break;
            clock_gettime(CLOCK_MONOTONIC_RAW, &req_buf_calc_start_time);
        }
        call_top_stencil(icnt, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn, top_stream);
        call_bot_stencil(icnt, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn, bot_stream);
        call_mid_stencil(icnt, new_nprocs, new_rank, nx, ny, nz, mgn, dx, dy, dz, dt, kappa, f, fn, mid_stream);
        swap(&f, &fn);
        icnt++;
    }
    MPI_Barrier(CTCA_subcomm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &req_main_end_time);
    int message = 0;

    if (new_rank == 0)
        MPI_Recv(&message, 1, MPI_INT, new_nprocs + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    clock_gettime(CLOCK_MONOTONIC_RAW, &coupled_app_end_time);

    float coupled_app_ltime[1];
    coupled_app_ltime[0] = output_float_time(coupled_app_start_time, coupled_app_end_time);
    float coupled_app_max_gtime[1];
    reduce_max_gtime(coupled_app_ltime, coupled_app_max_gtime, 1, CTCA_subcomm);

    float req_main_ltime[1];
    req_main_ltime[0] = output_float_time(req_main_start_time, req_main_end_time);
    float req_main_max_gtime[1];
    reduce_max_gtime(req_main_ltime, req_main_max_gtime, 1, CTCA_subcomm);

    float req_ctca_ltime[1];
    req_ctca_ltime[0] = output_float_time(req_ctca_init_start_time, req_ctca_init_end_time);
    float req_ctca_max_gtime[1];
    reduce_max_gtime(req_ctca_ltime, req_ctca_max_gtime, 1, CTCA_subcomm);

    // float *req_buf_write_max_ranktime;
    // req_buf_write_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    // reduce_max_gtime(req_buf_write_time_list, req_buf_write_max_ranktime, step_num, CTCA_subcomm);

    // float *req_buf_calc_max_ranktime;
    // req_buf_calc_max_ranktime = (float *)malloc(sizeof(float) * step_num - 1);
    // reduce_max_gtime(req_buf_calc_time_list, req_buf_calc_max_ranktime, step_num - 1, CTCA_subcomm);

    if (new_rank == 0)
    {
        float req_buf_write_max_gtime, req_buf_write_min_gtime, req_buf_write_avg_gtime;
        float req_buf_calc_max_gtime, req_buf_calc_min_gtime, req_buf_calc_avg_gtime;
        float req_packing_max_gtime, req_packing_min_gtime, req_packing_avg_gtime;
        float req_reshape_max_gtime, req_reshape_min_gtime, req_reshape_avg_gtime;
        float req_polling_max_gtime, req_polling_min_gtime, req_polling_avg_gtime;
        float req_enq_max_gtime, req_enq_min_gtime, req_enq_avg_gtime;

        calc_measure_time(req_buf_write_time_list, step_num, &req_buf_write_avg_gtime, &req_buf_write_max_gtime, &req_buf_write_min_gtime);
        calc_measure_time(req_buf_calc_time_list, step_num - 1, &req_buf_calc_avg_gtime, &req_buf_calc_max_gtime, &req_buf_calc_min_gtime);
        calc_measure_time(req_packing_time_list, step_num, &req_packing_avg_gtime, &req_packing_max_gtime, &req_packing_min_gtime);
        calc_measure_time(req_reshape_time_list, step_num, &req_reshape_avg_gtime, &req_reshape_max_gtime, &req_reshape_min_gtime);
        calc_measure_time(req_polling_time_list, step_num, &req_polling_avg_gtime, &req_polling_max_gtime, &req_polling_min_gtime);
        calc_measure_time(req_enq_time_list, step_num, &req_enq_avg_gtime, &req_enq_max_gtime, &req_enq_min_gtime);

        printf("coupled_global_time: max_time_per_process=%f sec\n", coupled_app_max_gtime[0]);
        printf("Sim main_global_time: max_time_per_process=%f sec\n", req_main_max_gtime[0]);
        printf("Sim write_global_time: ave=%f, max=%f, min=%f sec\n", req_buf_write_avg_gtime, req_buf_write_max_gtime, req_buf_write_min_gtime);
        printf("Sim packing_global_time: ave=%f, max=%f, min=%f sec\n", req_buf_write_avg_gtime, req_buf_write_max_gtime, req_buf_write_min_gtime);
        printf("Sim reshape_global_time: ave=%f, max=%f, min=%f sec\n", req_reshape_avg_gtime, req_reshape_max_gtime, req_reshape_min_gtime);
        printf("Sim polling_global_time: ave=%f, max=%f, min=%f sec\n", req_polling_avg_gtime, req_polling_max_gtime, req_polling_min_gtime);
        printf("Sim enq_global_time: ave=%f, max=%f, min=%f sec\n", req_enq_avg_gtime, req_enq_max_gtime, req_enq_min_gtime);
        printf("Sim calc_global_time: ave=%f, max=%f, min=%f sec\n", req_buf_calc_avg_gtime, req_buf_calc_max_gtime, req_buf_calc_min_gtime);
        printf("(Sim ctca_init_global_time: max_time_per_process=%f sec)\n", req_ctca_max_gtime[0]);
    }
    cudaStreamDestroy(top_stream);
    cudaStreamDestroy(bot_stream);
    cudaStreamDestroy(mid_stream);
    cudaStreamDestroy(packing_stream);

    cudaFree(f);
    f = NULL;
    cudaFree(fn);
    fn = NULL;
    cudaFree(y_axis_normal_plane_perprocs);
    y_axis_normal_plane_perprocs = NULL;
    cudaFree(y_axis_normal_plane_allprocs);
    y_axis_normal_plane_allprocs = NULL;
    free(reverse_slice_f);
    reverse_slice_f = NULL;
    free(result);
    result = NULL;
    free(f_H);
    f_H = NULL;
    CTCAR_finalize();

    return 0;
}

void write_to_file(const float *data, int num, int lenx, int lenz, int rank, int y, int step)
{
    char filename[100];
    sprintf(filename, "./test_step%d/rank%d_y%d.csv", step, rank, y); // ファイル名を作成

    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        return;
    }

    // num * num のデータを横軸 num、縦軸 num として書き込む
    for (int i = 0; i < lenz; i++)
    {
        for (int j = 0; j < lenx; j++)
        {
            if (j != 0)
            {
                fprintf(file, ",");
            }
            fprintf(file, "%.3f", data[i * lenx + j]); // 横軸 i、縦軸 j のデータ
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void initialize_con_packing_count(int *con_packing_count, int num)
{
    if (num <= CON_PACKING_COUNT_MAX)
    {
        *con_packing_count = num;
    }
    else if (num % CON_PACKING_COUNT_MAX == 0)
    {
        *con_packing_count = CON_PACKING_COUNT_MAX;
    }
    else
    {
        printf("num%d divided con_packing_count%d is not 0, so abort\n", num, CON_PACKING_COUNT_MAX);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}