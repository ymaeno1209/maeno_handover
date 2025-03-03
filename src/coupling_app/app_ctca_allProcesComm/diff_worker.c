#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>

#include "myPng.h"
#include "ctca.h"
#include "util.h"

#define PI 3.14
#define BUFFER_SIZE 1024 * 1024
#define MAX_ELEMENTS 1024 * 1024 // 配列の最大要素数
void write_to_file(const float *data, int num, int lenx, int lenz, int rank, int y, int step);
int main(int argc, char *argv[])
{
    struct timeval tv0_main;
    gettimeofday(&tv0_main, NULL);

    struct timeval tv1_main;
    const int num = atoi(argv[1]);
    const int step_num = atoi(argv[2]);
    const int Px = atoi(argv[3]);
    const int Pz = atoi(argv[4]);
    const int progid = 0;
    const int procs_per_subcomm = atoi(argv[5]);
    const int node = atoi(argv[6]);
    const int simppr = atoi(argv[7]);
    const int visppr = atoi(argv[8]);
    const int calctimes = atoi(argv[9]);
    const int lenx = num / Px;
    const int lenz = num / Pz;
    if (lenx * Px != num)
        printf("num=%dはPx=%dで割り切れません\n", num, Px);
    if (lenz * Pz != num)
        printf("num=%dはPz=%dで割り切れません\n", num, Pz);

    int new_nprocs, new_rank;

    struct timespec wrk_main_start_time, wrk_main_end_time;
    struct timespec wrk_buf_read_start_time, wrk_buf_read_end_time;
    struct timespec wrk_buf_rgb_start_time, wrk_buf_rgb_end_time;
    struct timespec wrk_buf_rgbcomm_start_time, wrk_buf_rgbcomm_end_time;
    struct timespec wrk_buf_fileIO_start_time, wrk_buf_fileIO_end_time;
    struct timespec wrk_polling_start_time, wrk_polling_end_time;
    struct timespec wrk_readarea_start_time, wrk_readarea_end_time;
    struct timespec wrk_scatter_start_time, wrk_scatter_end_time;
    struct timespec wrk_complete_start_time, wrk_complete_end_time;
    struct timespec wrk_packing_start_time, wrk_packing_end_time;
    struct timespec wrk_tmp_start_time, wrk_tmp_end_time;

    float *wrk_buf_read_time_list;
    float *wrk_buf_rgb_time_list;
    float *wrk_buf_rgbcomm_time_list;
    float *wrk_buf_fileIO_time_list;
    float *wrk_polling_time_list;
    float *wrk_readarea_time_list;
    float *wrk_scatter_time_list;
    float *wrk_complete_time_list;
    float *wrk_packing_time_list;
    float *wrk_tmp_time_list;
    wrk_buf_read_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_buf_rgb_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_buf_rgbcomm_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_buf_fileIO_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_polling_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_readarea_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_scatter_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_complete_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_packing_time_list = (float *)malloc(step_num * sizeof(float));
    wrk_tmp_time_list = (float *)malloc(num * step_num * sizeof(float));

    for (int i = 0; i < step_num; i++)
    {
        wrk_polling_time_list[i] = 0.0;
        wrk_readarea_time_list[i] = 0.0;
        wrk_scatter_time_list[i] = 0.0;
        wrk_complete_time_list[i] = 0.0;
        wrk_packing_time_list[i] = 0.0;
    }
    for (int i = 0; i < num * step_num; i++)
    {
        wrk_tmp_time_list[i] = 0.0;
    }

    // CTCAW_init_detail(progid, procs_per_subcomm, 1000, 1000);
    CTCAW_init(progid, procs_per_subcomm);

    MPI_Comm_size(CTCA_subcomm, &new_nprocs);
    MPI_Comm_rank(CTCA_subcomm, &new_rank);
    float *y_axis_normal_plane_allprocs_H = (float *)malloc(sizeof(float) * num * num * num);
    float *y_axis_normal_plane_everyprocs_H = (float *)malloc(sizeof(float) * lenx * lenz * num);
    memset(y_axis_normal_plane_everyprocs_H, 0, sizeof(y_axis_normal_plane_allprocs_H));
    float *packing_tmp_buffer = (float *)malloc(sizeof(float) * num * num * num);
    int every_procs_print_y_element = num / new_nprocs;

    char hostname[128];
    gethostname(hostname, sizeof(hostname));
    printf("vis : new_rank %d: hostname = %s\n", new_rank, hostname);
    BITMAPDATA_t *bitmap_print = (BITMAPDATA_t *)malloc((size_t)num / (size_t)new_nprocs * (size_t)sizeof(BITMAPDATA_t));
    if (bitmap_print == NULL)
    {
        printf("bitmap_print malloc error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    BITMAPDATA_t *bitmap_buf = (BITMAPDATA_t *)malloc((size_t)num * (size_t)sizeof(BITMAPDATA_t));
    if (bitmap_buf == NULL)
    {
        printf("bitmap_buf malloc error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* 作成するビットマップデータの情報格納 */
    for (int i = 0; i < every_procs_print_y_element; i++)
    {
        bitmap_print[i].width = num + 16;
        bitmap_print[i].height = num + 16;
        bitmap_print[i].ch = 3;

        /* ビットマップデータのメモリ確保 */
        bitmap_print[i].data = (unsigned char *)malloc(sizeof(unsigned char) * bitmap_print[i].width * bitmap_print[i].height * bitmap_print[i].ch);
        if (bitmap_print[i].data == NULL)
        {
            printf("bitmap_print[%d].data malloc error\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* ビットマップデータの背景を白にする */
        memset(bitmap_print[i].data, 0xFF, bitmap_print[i].width * bitmap_print[i].height * bitmap_print[i].ch); // 端の色のみ初期化する方が高速だが保留
    }
    for (int i = 0; i < num; i++)
    {
        bitmap_buf[i].width = lenx;
        bitmap_buf[i].height = lenz;
        bitmap_buf[i].ch = 3;

        /* ビットマップデータのメモリ確保 */
        bitmap_buf[i].data = (unsigned char *)malloc(sizeof(unsigned char) * bitmap_buf[i].width * bitmap_buf[i].height * bitmap_buf[i].ch);
        if (bitmap_buf[i].data == NULL)
        {
            printf("bitmap_buf[%d].data malloc error\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int squ_num = pow(num, 2);
    int a_block_num = lenx * lenz;
    float *print_buf;
    int intparams[2];
    int i, j, rank_Px, rank_Pz, fromrank, areaid;
    const int vis_x = 8;
    const int vis_y = 8;
    const size_t whole_surface_num = (size_t)num * (size_t)num;
    const int numintparams = 2;

    print_buf = (float *)malloc(a_block_num * sizeof(float));
    rank_Px = new_rank % Px;
    rank_Pz = new_rank / Px;
    int step = 0;
    int dest_rank_Px, dest_rank_Pz;

    CTCAW_regarea_real4(&areaid);
    for (int i = 0; i < lenz * 3; i++)
    {
        CTCAW_readarea_real4(areaid, 0, lenz * i, lenx, y_axis_normal_plane_everyprocs_H);
    }
    size_t rgb_comm_count = (size_t)(num - every_procs_print_y_element) * (size_t)lenz * 2L;
    MPI_Request *req = (MPI_Request *)malloc(rgb_comm_count * sizeof(MPI_Request));
    MPI_Request *scatter_req = (MPI_Request *)malloc(num * sizeof(MPI_Request));
    for (size_t i = 0; i < rgb_comm_count; i++)
    {
        req[i] = MPI_REQUEST_NULL;
    }
    for (int i = 0; i < num; i++)
    {
        scatter_req[i] = MPI_REQUEST_NULL;
    }
    MPI_Status *stat = (MPI_Status *)malloc(rgb_comm_count * sizeof(MPI_Status));
    MPI_Status *scatter_stat = (MPI_Status *)malloc(num * sizeof(MPI_Status));

    // debug
    double prev_max_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(CTCA_subcomm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_main_start_time);
    while (1)
    {
        MPI_Barrier(CTCA_subcomm);
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_read_start_time);

        for (int y = 0; y < num; y++)
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_polling_start_time);
            CTCAW_pollreq(&fromrank, intparams, numintparams);
            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_polling_end_time);
            wrk_polling_time_list[step] += output_float_time(wrk_polling_start_time, wrk_polling_end_time);

            if (CTCAW_isfin())
                break;
            MPI_Barrier(CTCA_subcomm);
            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_readarea_start_time);
            // printf("new_rank=%d,lenz=%d\n", new_rank, lenz);
            for (int z = 0; z < lenz; z++)
            {
                size_t dest_addr = z * lenx + y * lenz * lenx;
                size_t src_offset = (size_t)intparams[1] * (size_t)whole_surface_num + (size_t)rank_Px * (size_t)lenx + (size_t)rank_Pz * (size_t)num * (size_t)lenz + (size_t)z * (size_t)num;
                // if (new_rank == 0)
                //     printf("dest_addr=%lu, src_addr=%lu, new_rank=%d, step=%d, lenx=%d,y=%d, intparams[1]=%d\n", dest_addr, src_addr, new_rank, step, lenx, y, intparams[1]);
                clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_tmp_start_time);

                CTCAW_readarea_real4(areaid, fromrank, src_offset, lenx, &y_axis_normal_plane_everyprocs_H[dest_addr]);
                // if (new_rank == 0 && step == 0)
                //     printf("new_rank=%d,y=%d,z=%d,wrk_tmp = %f\n", new_rank, y, z, output_float_time(wrk_tmp_start_time, wrk_tmp_end_time));
                // if (new_rank == 0)
                //     printf("step=%d, z=%d,y=%d, output_float_time=%f\n", step, z, y, output_float_time(wrk_tmp_start_time, wrk_tmp_end_time));
                clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_tmp_end_time);
                wrk_tmp_time_list[y + step * num] += output_float_time(wrk_tmp_start_time, wrk_tmp_end_time);
            }

            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_readarea_end_time);
            wrk_readarea_time_list[step] += output_float_time(wrk_readarea_start_time, wrk_readarea_end_time);

            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_packing_start_time);

            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_packing_end_time);
            wrk_packing_time_list[step] += output_float_time(wrk_packing_start_time, wrk_packing_end_time);

            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_complete_start_time);
            CTCAW_complete();
            clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_complete_end_time);
            wrk_complete_time_list[step] += output_float_time(wrk_complete_start_time, wrk_complete_end_time);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_scatter_start_time);
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_scatter_end_time);
        wrk_scatter_time_list[step] = output_float_time(wrk_scatter_start_time, wrk_scatter_end_time);
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_read_end_time);
        wrk_buf_read_time_list[step] = output_float_time(wrk_buf_read_start_time, wrk_buf_read_end_time);

        /* (30, 30)から点を描画 */
        // #pragma omp parallel for private(i,j)、時間がかかるためコメントアウトした。
        MPI_Barrier(CTCA_subcomm);
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_rgb_start_time);
        for (int y = 0; y < num; y++)
        {
            for (int z = 0; z < lenz; z++)
            {
                for (int x = 0; x < lenx; x++)
                {
                    drawDot(
                        bitmap_buf[y].data,
                        bitmap_buf[y].width,
                        bitmap_buf[y].height,
                        x, z, /* 点を描画する座標 */
                        y_axis_normal_plane_everyprocs_H[y * lenx * lenz + z * lenx + x]);
                }
            }
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_rgb_end_time);
        wrk_buf_rgb_time_list[step] = output_float_time(wrk_buf_rgb_start_time, wrk_buf_rgb_end_time);

        // rgbの値を全てnew_rank=0に集める
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_rgbcomm_start_time);
        rgb_comm_to_allprocs(num, new_rank, new_nprocs, Px, lenx, lenz, vis_x, vis_y, rank_Px, rank_Pz, bitmap_buf, bitmap_print, CTCA_subcomm, req);
        MPI_Waitall(rgb_comm_count, req, stat);
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_rgbcomm_end_time);
        wrk_buf_rgbcomm_time_list[step] = output_float_time(wrk_buf_rgbcomm_start_time, wrk_buf_rgbcomm_end_time);

        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_fileIO_start_time);
        for (int i = 0; i < every_procs_print_y_element; i++)
        {
            int y = new_rank * every_procs_print_y_element + i;
            char filename[100];
            sprintf(filename, "num%d_node%d_simppr%d_visppr%d_stepnum%d_calctimes%d/step%d/y_%d.png", num, node, simppr, visppr, step_num, calctimes, step, y);
            if (pngFileEncodeWrite(&bitmap_print[i], filename) == -1)
            {
                freeBitmapData(&bitmap_print[i]);
                return -1;
            }
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_buf_fileIO_end_time);
        wrk_buf_fileIO_time_list[step] = output_float_time(wrk_buf_fileIO_start_time, wrk_buf_fileIO_end_time);

        step++;
        if (step == step_num)
            break;
    }
    MPI_Barrier(CTCA_subcomm);
    clock_gettime(CLOCK_MONOTONIC_RAW, &wrk_main_end_time);
    float wrk_main_ltime[1];
    wrk_main_ltime[0] = output_float_time(wrk_main_start_time, wrk_main_end_time);
    int message = 0;
    if (new_rank == 0)
        MPI_Send(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    float wrk_main_max_gtime[1];
    reduce_max_gtime(wrk_main_ltime, wrk_main_max_gtime, 1, CTCA_subcomm);

    float *wrk_buf_read_max_ranktime;
    wrk_buf_read_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    reduce_max_gtime(wrk_buf_read_time_list, wrk_buf_read_max_ranktime, step_num, CTCA_subcomm);

    float *wrk_buf_rgb_max_ranktime;
    wrk_buf_rgb_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    reduce_max_gtime(wrk_buf_rgb_time_list, wrk_buf_rgb_max_ranktime, step_num, CTCA_subcomm);

    float wrk_buf_rgbcomm_ave_gtime, wrk_buf_rgbcomm_max_gtime, wrk_buf_rgbcomm_min_gtime;
    float *wrk_buf_rgbcomm_max_ranktime;
    wrk_buf_rgbcomm_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    reduce_max_gtime(wrk_buf_rgbcomm_time_list, wrk_buf_rgbcomm_max_ranktime, step_num, CTCA_subcomm);

    float *wrk_buf_fileIO_max_ranktime;
    wrk_buf_fileIO_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    reduce_max_gtime(wrk_buf_fileIO_time_list, wrk_buf_fileIO_max_ranktime, step_num, CTCA_subcomm);

    float *wrk_readarea_max_ranktime;
    wrk_readarea_max_ranktime = (float *)malloc(sizeof(float) * step_num);
    reduce_max_gtime(wrk_readarea_time_list, wrk_readarea_max_ranktime, step_num, CTCA_subcomm);

    if (new_rank == 0)
    {
        for (int i = 0; i < step_num; i++)
        {
            for (int j = 0; j < num; j++)
                printf("wrk_tmp_time_list[%d]=%f\n", i * num + j, wrk_tmp_time_list[i * num + j]);
        }
    }

    for (int i = 0; i < step_num; i++)
    {
        printf("new_rank=%d, wrk_readarea_time_list[%d]=%f\n", new_rank, i, wrk_readarea_time_list[i]);
    }
    if (new_rank == 0)
    {
        float wrk_buf_read_ave_gtime, wrk_buf_read_max_gtime, wrk_buf_read_min_gtime;
        float wrk_buf_rgb_ave_gtime, wrk_buf_rgb_max_gtime, wrk_buf_rgb_min_gtime;
        float wrk_buf_fileIO_ave_gtime, wrk_buf_fileIO_max_gtime, wrk_buf_fileIO_min_gtime;
        float wrk_polling_ave_gtime, wrk_polling_max_gtime, wrk_polling_min_gtime;
        float wrk_readarea_ave_time, wrk_readarea_max_gtime, wrk_readarea_min_gtime;
        float wrk_scatter_ave_time, wrk_scatter_max_gtime, wrk_scatter_min_gtime;
        float wrk_complete_ave_time, wrk_complete_max_gtime, wrk_complete_min_gtime;
        float wrk_packing_ave_time, wrk_packing_max_gtime, wrk_packing_min_gtime;

        calc_measure_time(wrk_buf_read_max_ranktime, step_num, &wrk_buf_read_ave_gtime, &wrk_buf_read_max_gtime, &wrk_buf_read_min_gtime);
        calc_measure_time(wrk_buf_rgb_max_ranktime, step_num, &wrk_buf_rgb_ave_gtime, &wrk_buf_rgb_max_gtime, &wrk_buf_rgb_min_gtime);
        calc_measure_time(wrk_buf_rgbcomm_max_ranktime, step_num, &wrk_buf_rgbcomm_ave_gtime, &wrk_buf_rgbcomm_max_gtime, &wrk_buf_rgbcomm_min_gtime);
        calc_measure_time(wrk_buf_fileIO_time_list, step_num, &wrk_buf_fileIO_ave_gtime, &wrk_buf_fileIO_max_gtime, &wrk_buf_fileIO_min_gtime);
        calc_measure_time(wrk_polling_time_list, step_num, &wrk_polling_ave_gtime, &wrk_polling_max_gtime, &wrk_polling_min_gtime);
        calc_measure_time(wrk_readarea_max_ranktime, step_num, &wrk_readarea_ave_time, &wrk_readarea_max_gtime, &wrk_readarea_min_gtime);
        calc_measure_time(wrk_scatter_time_list, step_num, &wrk_scatter_ave_time, &wrk_scatter_max_gtime, &wrk_scatter_min_gtime);
        calc_measure_time(wrk_complete_time_list, step_num, &wrk_complete_ave_time, &wrk_complete_max_gtime, &wrk_complete_min_gtime);
        calc_measure_time(wrk_packing_time_list, step_num, &wrk_packing_ave_time, &wrk_packing_max_gtime, &wrk_packing_min_gtime);

        printf("Vis main_global_time: max_time_per_process=%f sec\n", wrk_main_max_gtime[0]);
        printf("Vis read_global_time: ave=%f, max=%f, min=%f sec\n", wrk_buf_read_ave_gtime, wrk_buf_read_max_gtime, wrk_buf_read_min_gtime);
        printf("Vis polling_global_time: ave=%f, max=%f, min=%f sec\n", wrk_polling_ave_gtime, wrk_polling_max_gtime, wrk_polling_min_gtime);
        printf("Vis readarea_global_time: ave=%f, max=%f, min=%f sec\n", wrk_readarea_ave_time, wrk_readarea_max_gtime, wrk_readarea_min_gtime);
        printf("Vis packing_global_time: ave=%f, max=%f, min=%f sec\n", wrk_packing_ave_time, wrk_packing_max_gtime, wrk_packing_min_gtime);
        printf("Vis complete_global_time: ave=%f, max=%f, min=%f sec\n", wrk_complete_ave_time, wrk_complete_max_gtime, wrk_complete_min_gtime);
        printf("Vis scatter_global_time: ave=%f, max=%f, min=%f sec\n", wrk_scatter_ave_time, wrk_scatter_max_gtime, wrk_scatter_min_gtime);
        printf("Vis rgb_global_time: ave=%f, max=%f, min=%f sec\n", wrk_buf_rgb_ave_gtime, wrk_buf_rgb_max_gtime, wrk_buf_rgb_min_gtime);
        printf("Vis rgbcomm_global_time: ave=%f, max=%f, min=%f sec\n", wrk_buf_rgbcomm_ave_gtime, wrk_buf_rgbcomm_max_gtime, wrk_buf_rgbcomm_min_gtime);
        printf("Vis fileIO_global_time: ave=%f, max=%f, min=%f sec\n", wrk_buf_fileIO_ave_gtime, wrk_buf_fileIO_max_gtime, wrk_buf_fileIO_min_gtime);
    }

    free(req);
    free(stat);
    req = NULL;
    stat = NULL;

    for (int i = 0; i < num; i++)
    {
        freeBitmapData(&bitmap_buf[i]);
    }
    for (int i = 0; i < every_procs_print_y_element; i++)
    {
        freeBitmapData(&bitmap_print[i]);
    }
    if (bitmap_buf != NULL && bitmap_buf != NULL)
    {
        free(bitmap_buf);
        free(bitmap_print);
    }

    CTCAW_finalize();

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