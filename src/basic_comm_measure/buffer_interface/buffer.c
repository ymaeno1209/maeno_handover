#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <locale.h>
#include <time.h>
#include <sys/time.h>
#include "ctca.h"
#include "buffer.h"
#include "buffer_kernel.h"

#ifdef _ENABLE_CUDA_
#include <cuda_runtime.h>
#endif

MPI_Comm CTCA_subcomm;

// measure_functions
float buffer_output_float_time(struct timespec start_time, struct timespec end_time);
void buffer_calc_measure_time(float *elapsed_time, int iteration, float *average_time, float *max_time, float *min_time);
float buffer_reduce_max_gtime(float *ltime_list, float *max_gtime, int step_num, MPI_Comm new_comm);

#define MAX_BUFFER_SLOT 4 // 元々は4
#define GAREAID_ELEMENT 2
#define DATA_DIVISION_NUM 3
#define WORKER_INIT_INFO_ELEMENT 4
#define REQUESTER_INIT_INFO_ELEMENT 1
#define INIT_INFO_INITIAL -1
#define INITIAL -1
#define HEADER_INITIAL -1
#define REAL_BUFFER_INITIAL -1
#define DATA_INT 0
#define DATA_REAL4 1
#define DATA_REAL8 2
#define ROOT_RANK 0
#define XY_INT 0
#define YZ_INT 1
#define ZX_INT 2
#define MAX_TARGET_PROCS 10
#define MAX_COMM_TIMES ((size_t)1024 * 1024 * 1024)
#define MAX_PACKING_INDEX ((size_t)1024 * 1024 * 1024)
#define PACKED_DATA_ELEMENT_INITIAL 0
#define PACKED_DATA_BYTE_INITIAL 0
#define HEADER_CHANK_ELEMENT 1
#define WIDTH_ELEMENT_INITIAL 0
#define HEIGHT_ELEMENT_INITIAL 0
#define MAX_MEASURE_ELEMENT 1024 * 1024
#define CON_PACKING_COUNT_MAX 128

static int *worker_init_info;
static int *requester_init_info;
static int init_info_gareaid;
static int real_buffer_gareaid;
static int header_buffer_gareaid;
static void *real_buffer;
static int *header_buffer;
static int submyrank;
static int subnprocs;
static int world_myrank;
static int world_nprocs;
static int data_type;
static int unitsize;
static int requester_root_world_rank;
static int worker_root_world_rank;
static size_t data_oneside_length;
static int worker_procs_x;
static int worker_procs_y;
static int worker_procs_z;
static int requester_procs_x;
static int requester_procs_y;
static int requester_procs_z;
static int requester_element_x;
static int requester_element_y;
static int requester_element_z;
static int worker_element_x;
static int worker_element_y;
static int worker_element_z;
static size_t real_buffer_oneslot_byte;
static size_t real_buffer_oneslot_element;
static size_t real_buffer_oneslot_allprocs_element;
static void *comm_data0;
static void *comm_data1;
static struct requester_write_info_t *requester_write_info_list;
static size_t requester_write_times = 0L;
static struct timespec write_polling_start_time, write_polling_end_time;
static struct timespec write_start_time, write_end_time;
static struct timespec packing_start_time, packing_end_time;
static struct timespec all_garea_write_start_time, all_garea_write_end_time;
static struct timespec read_polling_start_time, read_polling_end_time;
static struct timespec read_start_time, read_end_time;
static int write_time_ctr = 0;
static int read_time_ctr = 0;
static int packing_time_ctr = 0;
static int all_garea_write_time_ctr = 0;
static float write_polling_time_list[MAX_MEASURE_ELEMENT];
static float read_polling_time_list[MAX_MEASURE_ELEMENT];
static float packing_time_list[MAX_MEASURE_ELEMENT];
static float all_garea_write_time_list[MAX_MEASURE_ELEMENT];
static float write_time_list[MAX_MEASURE_ELEMENT];
static float read_time_list[MAX_MEASURE_ELEMENT];
static size_t *packed_data_byte;
static size_t *packed_data_element0;
static size_t *packed_data_element1;
static size_t *src_data_offset;
static size_t *target_data_offset0;
static size_t *target_data_offset1;
static size_t *height_element;
static size_t *width_element;
static size_t *width_byte;
static int *target_rank0;
static int *target_rank1;
static int buffer_index_counter[MAX_BUFFER_SLOT];
static size_t *write_info_D;
static size_t con_packing_count;
static int loop_ctr_max;
static size_t packed_data_byte_max = 0;
static size_t *write_info;
static int loop_packing_write_count = 0;

#ifdef _ENABLE_CUDA_
static cudaError_t err;
static cudaStream_t packing_stream;
static size_t *packed_data_byte_D, *src_data_offset_D, *width_byte_D;
#endif

void requester_buffer_init_withint(int *requester_data_division, size_t datasize, bool data_direction)
{
    requester_buffer_init(requester_data_division, datasize, data_direction, DATA_INT);
}

void requester_buffer_init_withreal4(int *requester_data_division, size_t datasize, bool data_direction)
{
    requester_buffer_init(requester_data_division, datasize, data_direction, DATA_REAL4);
}

void requester_buffer_init_withreal8(int *requester_data_division, size_t datasize, bool data_direction)
{
    requester_buffer_init(requester_data_division, datasize, data_direction, DATA_REAL8);
}

void worker_buffer_init_withint(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length)
{
    worker_buffer_init(worker_data_division, target_side, worker_data_oneside_length, DATA_INT);
}

void worker_buffer_init_withreal4(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length)
{
    worker_buffer_init(worker_data_division, target_side, worker_data_oneside_length, DATA_REAL4);
}

void worker_buffer_init_withreal8(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length)
{
    worker_buffer_init(worker_data_division, target_side, worker_data_oneside_length, DATA_REAL8);
}

void requester_buffer_init(int *requester_data_division, size_t datasize, bool data_direction, int type)
{
    int offset;
    int worker_wgid = 0;
    worker_init_info = (int *)malloc(WORKER_INIT_INFO_ELEMENT * sizeof(int));
    requester_init_info = (int *)malloc(REQUESTER_INIT_INFO_ELEMENT * sizeof(int));

    MPI_Comm_rank(CTCA_subcomm, &submyrank);
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);

    CTCAR_garea_create(&init_info_gareaid);
    CTCAR_garea_create(&header_buffer_gareaid);
    CTCAR_garea_create(&real_buffer_gareaid);

    data_type = type;
    unitsize = decide_unitsize(data_type);
    data_oneside_length = (size_t)round(pow((double)datasize, 1.0 / 3.0));
    requester_init_info[0] = data_oneside_length;
    memset(worker_init_info, INIT_INFO_INITIAL, WORKER_INIT_INFO_ELEMENT * sizeof(int));
    if (submyrank == ROOT_RANK)
    {
        offset = 0;
        CTCAR_garea_attach(init_info_gareaid, worker_init_info, WORKER_INIT_INFO_ELEMENT * sizeof(int));
        CTCAR_get_grank_wrk(worker_wgid, ROOT_RANK, &worker_root_world_rank);
        fetch_worker_init_info(worker_init_info);
        CTCAR_garea_write_int(init_info_gareaid, worker_root_world_rank, offset, REQUESTER_INIT_INFO_ELEMENT, requester_init_info);
    }
    MPI_Bcast(worker_init_info, WORKER_INIT_INFO_ELEMENT, MPI_INT, ROOT_RANK, CTCA_subcomm);
    setup_conditions(worker_init_info, requester_data_division);
    initialize_requester_write_info();
    initialize_con_packing_count();
    reshape_requester_write_info();
    initialize_count_buffer_index();
}

void requester_buffer_write(void *src_data, int step)
{
    int buffer_index;
    clock_gettime(CLOCK_MONOTONIC, &write_polling_start_time);
    buffer_index = requester_polling_header_buffer();
    clock_gettime(CLOCK_MONOTONIC, &write_polling_end_time);
    write_polling_time_list[write_time_ctr] = buffer_output_float_time(write_polling_start_time, write_polling_end_time);
    buffer_index_counter[buffer_index]++;

    clock_gettime(CLOCK_MONOTONIC, &write_start_time);
    requester_write_onestep_to_realbuffer(src_data, buffer_index);
    clock_gettime(CLOCK_MONOTONIC, &write_end_time);
    write_time_list[write_time_ctr] = buffer_output_float_time(write_start_time, write_end_time);

    MPI_Barrier(CTCA_subcomm); // 仮として同期、終了フラグをPutにした方が速いかも
    write_header_buffer(buffer_index, step);
    write_time_ctr++;
}

void requester_buffer_fin()
{
    MPI_Barrier(MPI_COMM_WORLD);
    CTCAR_garea_detach(init_info_gareaid, worker_init_info);
    CTCAR_garea_delete();
    free(real_buffer);
    free(header_buffer);
    free(worker_init_info);
    free(requester_init_info);
    free(packed_data_byte);
    free(packed_data_element0);
    free(packed_data_element1);
    free(src_data_offset);
    free(target_data_offset0);
    free(target_data_offset1);
    free(height_element);
    free(width_element);
    free(width_byte);

#ifdef _ENABLE_CUDA_
    cudaFree(comm_data0);
    cudaFree(comm_data1);
    cudaFree(write_info_D);
    cudaStreamDestroy(packing_stream);
#else
    free(comm_data0);
#endif
}

void coupler_buffer_init()
{
    CTCAC_garea_create(&init_info_gareaid);
    CTCAC_garea_create(&header_buffer_gareaid);
    CTCAC_garea_create(&real_buffer_gareaid);

    MPI_Comm_rank(CTCA_subcomm, &submyrank);
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);
}

void coupler_buffer_fin()
{
    MPI_Barrier(MPI_COMM_WORLD);
    CTCAC_garea_delete();
}

void worker_buffer_init(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length, int type)
{
    size_t offset;
    worker_init_info = (int *)malloc(WORKER_INIT_INFO_ELEMENT * sizeof(int));
    requester_init_info = (int *)malloc(REQUESTER_INIT_INFO_ELEMENT * sizeof(int));

    MPI_Comm_rank(CTCA_subcomm, &submyrank);
    MPI_Comm_size(CTCA_subcomm, &subnprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);

    worker_init_info = worker_collect_init_info(worker_data_division, target_side);
    memset(requester_init_info, INIT_INFO_INITIAL, REQUESTER_INIT_INFO_ELEMENT * sizeof(int));

    data_type = type;
    unitsize = decide_unitsize(data_type);

    CTCAW_garea_create(&init_info_gareaid);
    CTCAW_garea_create(&header_buffer_gareaid);
    CTCAW_garea_create(&real_buffer_gareaid);

    if (submyrank == ROOT_RANK)
    {
        offset = 0;
        CTCAW_garea_attach(init_info_gareaid, requester_init_info, sizeof(size_t));
        CTCAW_get_grank_req(ROOT_RANK, &requester_root_world_rank);
        CTCAW_garea_write_int(init_info_gareaid, requester_root_world_rank, offset, WORKER_INIT_INFO_ELEMENT, worker_init_info);
        fetch_requester_init_info(requester_init_info);
    }
    MPI_Bcast(requester_init_info, REQUESTER_INIT_INFO_ELEMENT, MPI_INT, ROOT_RANK, CTCA_subcomm);
    data_oneside_length = requester_init_info[0];
    *worker_data_oneside_length = data_oneside_length;

    initialize_header_buffer();
    initialize_real_buffer();
}

void worker_buffer_read(void *dest_data, int step)
{

    int buffer_index;
    size_t real_buffer_offset;

    clock_gettime(CLOCK_MONOTONIC, &read_polling_start_time);
    buffer_index = worker_polling_header_buffer(step);
    clock_gettime(CLOCK_MONOTONIC, &read_polling_end_time);
    read_polling_time_list[read_time_ctr] = buffer_output_float_time(read_polling_start_time, read_polling_end_time);

    // real_bufferからdataを読み込み、dest_dataにデータを書き込む
    real_buffer_offset = (size_t)buffer_index * real_buffer_oneslot_byte;

    clock_gettime(CLOCK_MONOTONIC, &read_start_time);
    memcpy(dest_data, &((char *)real_buffer)[real_buffer_offset], real_buffer_oneslot_byte);
    clock_gettime(CLOCK_MONOTONIC, &read_end_time);
    read_time_list[read_time_ctr] = buffer_output_float_time(read_start_time, read_end_time);

    MPI_Barrier(CTCA_subcomm); // 終了フラグにした方が早いかも
    // header_bufferに-1を代入
    clear_head_buffer_at_index(buffer_index);
    read_time_ctr++;
}

void worker_buffer_fin()
{
    MPI_Barrier(MPI_COMM_WORLD);
    CTCAW_garea_detach(init_info_gareaid, requester_init_info);
    CTCAW_garea_detach(header_buffer_gareaid, header_buffer);
    CTCAW_garea_detach(real_buffer_gareaid, real_buffer);
    free(header_buffer);
    free(real_buffer);
    free(worker_init_info);
    free(requester_init_info);
    CTCAW_garea_delete();
}

int *worker_collect_init_info(int *worker_data_division, target_side_type target_side)
{
    int i;
    int *worker_init_info;
    worker_init_info = (int *)malloc(WORKER_INIT_INFO_ELEMENT * sizeof(int));
    for (i = 0; i < DATA_DIVISION_NUM; i++)
        worker_init_info[i] = worker_data_division[i];
    switch (target_side)
    {
    case XY:
        worker_init_info[4] = XY_INT;
        return worker_init_info;
    case YZ:
        worker_init_info[4] = YZ_INT;
        return worker_init_info;
    case ZX:
        worker_init_info[4] = ZX_INT;
        return worker_init_info;
    default:
        fprintf(stderr, "error: worker_buffer_init submyrank=%d: targetside is not XY or YZ or ZX\n", submyrank);
        MPI_Abort(MPI_COMM_WORLD, 0);
        return NULL;
    }
}
void fetch_worker_init_info(int *worker_init_info)
{
    int i;
    int init_info_ctr;
    while (1)
    {
        init_info_ctr = 0;
        for (i = 0; i < WORKER_INIT_INFO_ELEMENT; i++)
        {
            if (worker_init_info[i] > -1)
            {
                init_info_ctr++;
            }
        }
        if (init_info_ctr == WORKER_INIT_INFO_ELEMENT)
            break;
    }
}
void fetch_requester_init_info(int *requester_init_info)
{
    int i;
    int init_info_ctr;
    while (1)
    {
        init_info_ctr = 0;
        for (i = 0; i < REQUESTER_INIT_INFO_ELEMENT; i++)
        {
            if (requester_init_info[i] > -1)
            {
                init_info_ctr++;
            }
        }
        if (init_info_ctr == REQUESTER_INIT_INFO_ELEMENT)
            break;
    }
}

// to_worker_localrank_from_coordinateに問題有
int to_worker_localrank_from_coordinate(int global_x, int global_y, int global_z)
{
    int targetrank = 0;
    int worker_node_x;
    int worker_node_y;
    int worker_node_z;
    int i;
    for (i = 0; i < worker_procs_x; i++)
    {
        worker_node_x = (data_oneside_length / worker_procs_x) * (i + 1);
        if (global_x >= worker_node_x)
            targetrank += 1;
        else
            break;
    }
    for (i = 0; i < worker_procs_y; i++)
    {
        worker_node_y = (data_oneside_length / worker_procs_y) * (i + 1);
        if (global_y >= worker_node_y)
            targetrank += worker_procs_y;
        else
            break;
    }
    for (i = worker_procs_z - 1; i >= 0; i--)
    {
        worker_node_z = (data_oneside_length / worker_procs_z) * i;
        if (global_z < worker_node_z)
            targetrank += worker_procs_x * worker_procs_y;
        else
            break;
    }

    return targetrank;
}

size_t to_offset_from_coordinate(int global_x, int global_y, int global_z)
{
    size_t offset = (size_t)0;

    offset += (size_t)global_x % (size_t)worker_element_x;
    offset += (size_t)global_y * ((size_t)worker_element_x * (size_t)worker_element_z);
    offset += (((size_t)worker_element_z - (size_t)1) - ((size_t)global_z % (size_t)worker_element_z)) * (size_t)worker_element_z;
    return offset;
}

void setup_conditions(int *worker_init_info, int *requester_data_division)
{
    worker_procs_x = worker_init_info[0];
    worker_procs_y = worker_init_info[1];
    worker_procs_z = worker_init_info[2];
    requester_procs_x = requester_data_division[0];
    requester_procs_y = requester_data_division[1];
    requester_procs_z = requester_data_division[2];
    requester_element_x = data_oneside_length / requester_procs_x;
    requester_element_y = data_oneside_length / requester_procs_y;
    requester_element_z = data_oneside_length / requester_procs_z;
    worker_element_x = data_oneside_length / worker_procs_x;
    worker_element_y = data_oneside_length / worker_procs_y;
    worker_element_z = data_oneside_length / worker_procs_z;
    real_buffer_oneslot_allprocs_element = worker_element_x * worker_element_y * worker_element_z;
    size_t comm_data_max_byte = (size_t)MAX_COMM_TIMES * (size_t)unitsize;
    // size_t comm_data_max_byte = (size_t)1024 * 1024 * (size_t)unitsize;
    packed_data_byte = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    packed_data_element0 = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    packed_data_element1 = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    src_data_offset = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    target_data_offset0 = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    target_data_offset1 = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    height_element = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    width_element = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    width_byte = (size_t *)malloc(CON_PACKING_COUNT_MAX * sizeof(size_t));
    target_rank0 = (int *)malloc(CON_PACKING_COUNT_MAX * sizeof(int));
    target_rank1 = (int *)malloc(CON_PACKING_COUNT_MAX * sizeof(int));
#ifdef _ENABLE_CUDA_
    err = cudaMalloc((void **)&comm_data1, comm_data_max_byte);
    if (err != cudaSuccess)
    {
        printf("CUDA error:  cudaMalloc comm_data: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = cudaMalloc((void **)&comm_data0, comm_data_max_byte);
    if (err != cudaSuccess)
    {
        printf("CUDA error:  cudaMalloc comm_data: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cudaStreamCreate(&packing_stream);
#else
    comm_data0 = (void *)malloc(MAX_COMM_TIMES * unitsize);
#endif
}

void to_global_coordinate_from_local_coordinate_x(int local_x, int *global_x)
{
    *global_x = local_x; // x軸で分割する場合、コードの書き換え必須
}

void to_global_coordinate_from_local_coordinate_y(int local_y, int *global_y)
{
    *global_y = local_y; // y軸で分割する場合、コードの書き換え必須
}

void to_global_coordinate_from_local_coordinate_z(int local_z, int *global_z)
{
    *global_z = local_z + (submyrank * requester_element_z); // z軸以外で分割する場合変更必須
}

void garea_write_autotype(int real_buffer_gareaid, int *target_world_rank, size_t *offset, size_t *packed_data_element, void *comm_data, int con_packing_count)
{
    switch (data_type)
    {
    case DATA_INT:
        for (int i = 0; i < con_packing_count; i++)
        {
            CTCAR_garea_write_int(real_buffer_gareaid, target_world_rank[i], offset[i], packed_data_element[i], &((int *)comm_data)[i * packed_data_element[i]]);
        }
        break;
    case DATA_REAL4:
        for (int i = 0; i < con_packing_count; i++)
        {
            CTCAR_garea_write_real4(real_buffer_gareaid, target_world_rank[i], offset[i], packed_data_element[i], &((float *)comm_data)[i * packed_data_element[i]]);
        }
        break;
    case DATA_REAL8:
        for (int i = 0; i < con_packing_count; i++)
        {
            CTCAR_garea_write_real8(real_buffer_gareaid, target_world_rank[i], offset[i], packed_data_element[i], &((double *)comm_data)[i * packed_data_element[i]]);
        }
        break;
    default:
        fprintf(stderr, "%d : garea() : ERROR : wrong src_data type\n", world_myrank);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}

int requester_polling_header_buffer()
{
    int buffer_index_tmp;
    int header_step_tmp;
    int buffer_index;
    bool finding_flag = true;
    // 1回のreadで全スロットのheaderをGetし、そのデータからインデックス番号を探す方がいいかも
    if (submyrank == ROOT_RANK)
    {
        while (finding_flag)
        {
            for (buffer_index_tmp = 0; buffer_index_tmp < MAX_BUFFER_SLOT; buffer_index_tmp++)
            {
                CTCAR_garea_read_int(header_buffer_gareaid, worker_root_world_rank, buffer_index_tmp, HEADER_CHANK_ELEMENT, &header_step_tmp);
                if (header_step_tmp == HEADER_INITIAL)
                {
                    buffer_index = buffer_index_tmp;
                    finding_flag = false;
                    break;
                }
            }
        }
    }
    MPI_Bcast(&buffer_index, HEADER_CHANK_ELEMENT, MPI_INT, ROOT_RANK, CTCA_subcomm);
    return buffer_index;
}

int worker_polling_header_buffer(int step)
{
    int buffer_index_tmp;
    int header_step_tmp;
    int buffer_index;
    bool finding_flag = true;
    if (submyrank == ROOT_RANK)
    {
        while (finding_flag)
        {
            for (buffer_index_tmp = 0; buffer_index_tmp < MAX_BUFFER_SLOT; buffer_index_tmp++)
            {
                header_step_tmp = header_buffer[buffer_index_tmp];
                if (header_step_tmp == step)
                {
                    buffer_index = buffer_index_tmp;
                    finding_flag = false;
                    break;
                }
            }
        }
    }
    MPI_Bcast(&buffer_index, HEADER_CHANK_ELEMENT, MPI_INT, ROOT_RANK, CTCA_subcomm);
    return buffer_index;
}

void write_header_buffer(int buffer_index, int step)
{
    int header_buffer_offset;
    if (submyrank == ROOT_RANK)
    {
        header_buffer_offset = buffer_index; // 後で変更予定
        CTCAR_garea_write_int(header_buffer_gareaid, worker_root_world_rank, header_buffer_offset, HEADER_CHANK_ELEMENT, &step);
    }
}

void requester_write_onestep_to_realbuffer(void *src_data, int buffer_index)
{
    size_t square_byte = (size_t)data_oneside_length * (size_t)data_oneside_length * (size_t)unitsize; // data_oneside_length=num

#ifdef _ENABLE_CUDA_

    for (int i = 0; i < con_packing_count; i++)
    {
        height_element[i] = requester_write_info_list[i].height_element;
        width_element[i] = requester_write_info_list[i].width_element;
        packed_data_element0[i] = height_element[i] * width_element[i];
        target_data_offset0[i] = requester_write_info_list[i].dest_data_offset + (size_t)buffer_index * (size_t)real_buffer_oneslot_allprocs_element;
        target_rank0[i] = requester_write_info_list[i].target_rank;
    }
    call_con_pack_data(comm_data0, src_data, src_data_offset_D, packed_data_byte_D, packed_data_byte_max, width_byte_D, square_byte, con_packing_count, packing_stream);

    swap_data_addr(&comm_data0, &comm_data1);
    swap_data_real8(&target_data_offset0, &target_data_offset1);
    swap_data_real8(&packed_data_element0, &packed_data_element1);
    swap_data_int(&target_rank0, &target_rank1);
    for (int loop_ctr = 1; loop_ctr < loop_ctr_max; loop_ctr++)
    {

        for (int i = 0; i < con_packing_count; i++)
        {
            height_element[i] = requester_write_info_list[loop_ctr * con_packing_count + i].height_element;
            width_element[i] = requester_write_info_list[loop_ctr * con_packing_count + i].width_element;
            packed_data_element0[i] = height_element[i] * width_element[i];
            target_data_offset0[i] = requester_write_info_list[loop_ctr * con_packing_count + i].dest_data_offset + (size_t)buffer_index * (size_t)real_buffer_oneslot_allprocs_element;
            target_rank0[i] = requester_write_info_list[loop_ctr * con_packing_count + i].target_rank;
        }
        clock_gettime(CLOCK_MONOTONIC, &packing_start_time);
        cudaStreamSynchronize(packing_stream);
        clock_gettime(CLOCK_MONOTONIC, &packing_end_time);
        packing_time_list[packing_time_ctr] = buffer_output_float_time(packing_start_time, packing_end_time);
        packing_time_ctr++;

        call_con_pack_data(comm_data0, src_data, &src_data_offset_D[loop_ctr * con_packing_count], &packed_data_byte_D[loop_ctr * con_packing_count], packed_data_byte_max, &width_byte_D[loop_ctr * con_packing_count], square_byte, con_packing_count, packing_stream);
        clock_gettime(CLOCK_REALTIME, &all_garea_write_start_time);
        garea_write_autotype(real_buffer_gareaid, target_rank1, target_data_offset1, packed_data_element1, comm_data1, con_packing_count);
        clock_gettime(CLOCK_REALTIME, &all_garea_write_end_time);
        all_garea_write_time_list[all_garea_write_time_ctr] = buffer_output_float_time(all_garea_write_start_time, all_garea_write_end_time);
        all_garea_write_time_ctr++;
        swap_data_addr(&comm_data0, &comm_data1);
        swap_data_real8(&target_data_offset0, &target_data_offset1);
        swap_data_real8(&packed_data_element0, &packed_data_element1);
        swap_data_int(&target_rank0, &target_rank1);
    }
    clock_gettime(CLOCK_MONOTONIC, &packing_start_time);
    cudaStreamSynchronize(packing_stream);
    clock_gettime(CLOCK_MONOTONIC, &packing_end_time);
    packing_time_list[packing_time_ctr] = buffer_output_float_time(packing_start_time, packing_end_time);
    packing_time_ctr++;

    clock_gettime(CLOCK_REALTIME, &all_garea_write_start_time);
    garea_write_autotype(real_buffer_gareaid, target_rank1, target_data_offset1, packed_data_element1, comm_data1, con_packing_count);
    clock_gettime(CLOCK_REALTIME, &all_garea_write_end_time);
    all_garea_write_time_list[all_garea_write_time_ctr] = buffer_output_float_time(all_garea_write_start_time, all_garea_write_end_time);
    all_garea_write_time_ctr++;
#else // CPU用に用意したが、正常に動作しない

#endif
}

void initialize_header_buffer()
{
    size_t header_buffer_byte;

    header_buffer_byte = (size_t)MAX_BUFFER_SLOT * sizeof(int);
    header_buffer = (int *)malloc(header_buffer_byte);
    memset(header_buffer, HEADER_INITIAL, header_buffer_byte);
    CTCAW_garea_attach(header_buffer_gareaid, header_buffer, header_buffer_byte);
}

/**
 * @brief Initializes the real buffer for data storage.
 *
 * This function calculates the size of the real buffer based on the data dimensions and number of processes.
 * It then allocates memory for the real buffer, sets its initial values, and attaches it to a global area.
 * If the data cannot be split according to the specified division, an error message is printed and the program is aborted.
 */
void initialize_real_buffer()
{
    size_t real_buffer_byte;
    size_t tmp = data_oneside_length * data_oneside_length;
    tmp *= data_oneside_length;
    real_buffer_oneslot_element = tmp / (size_t)subnprocs;

    real_buffer_oneslot_byte = real_buffer_oneslot_element * (size_t)unitsize;

    real_buffer_byte = (size_t)MAX_BUFFER_SLOT * real_buffer_oneslot_byte;
    if (data_oneside_length * data_oneside_length * data_oneside_length % subnprocs != 0)
    {
        fprintf(stderr, "wrk%d error: Unable to split the requester src_data(data_oneside_length=%zu) according to the specified src_data division in the worker.\n",
                submyrank, data_oneside_length);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    real_buffer = (void *)malloc(real_buffer_byte);
    if (real_buffer == NULL)
    {
        fprintf(stderr, "error: Memory allocation failed for real_buffer.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(real_buffer, HEADER_INITIAL, real_buffer_byte);
    CTCAW_garea_attach(real_buffer_gareaid, real_buffer, real_buffer_byte);
}

void clear_head_buffer_at_index(int buffer_index)
{
    header_buffer[buffer_index] = HEADER_INITIAL;
}

void initialize_requester_write_info()
{
    int global_x, global_y, global_z;
    int local_x, local_y, local_z;
    size_t data_index;
    size_t packed_data_byte = (size_t)PACKED_DATA_ELEMENT_INITIAL;
    size_t data_index_byte;
    size_t data_onepack_byte;
    int target_world_rank;
    size_t real_buffer_offset = (size_t)INITIAL;
    int target_local_rank_pre = INITIAL;
    int target_local_rank_current = INITIAL;
    size_t packed_data_element = (size_t)PACKED_DATA_ELEMENT_INITIAL;
    size_t width_element = (size_t)worker_element_z;
    size_t height_element = (size_t)HEIGHT_ELEMENT_INITIAL;
    int worker_wgid = 0;
    size_t data_onepack_element = (size_t)worker_element_x;
    requester_write_info_list = (requester_write_info_t *)malloc(sizeof(requester_write_info_t) * MAX_COMM_TIMES);
    size_t req_packing_index;

    for (local_y = 0; local_y < requester_element_y; local_y++)
    {
        to_global_coordinate_from_local_coordinate_y(local_y, &global_y);
        for (local_x = 0; local_x < requester_element_x; local_x++)
        {
            to_global_coordinate_from_local_coordinate_x(local_x, &global_x);
            if (global_x % worker_element_x != 0)
                continue;
            for (local_z = requester_element_z - 1; local_z >= 0; local_z--)
            {
                to_global_coordinate_from_local_coordinate_z(local_z, &global_z);

                target_local_rank_current = to_worker_localrank_from_coordinate(global_x, global_y, global_z);

                if (target_local_rank_pre != target_local_rank_current && target_local_rank_pre != INITIAL)
                {
                    CTCAR_get_grank_wrk(worker_wgid, target_local_rank_pre, &target_world_rank);

                    requester_write_info_list[requester_write_times] = (requester_write_info_t){
                        .src_data_offset = req_packing_index,
                        .width_element = width_element,
                        .height_element = height_element,
                        .dest_data_offset = real_buffer_offset,
                        .target_rank = target_world_rank};

                    requester_write_times++;
                    if (requester_write_times >= MAX_COMM_TIMES)
                    {
                        fprintf(stderr, "Exceeded MAX_COMM_TIMES in initialize_requester_write_info\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    // 後処理
                    target_local_rank_pre = INITIAL;
                    packed_data_element = (size_t)PACKED_DATA_ELEMENT_INITIAL;
                    packed_data_byte = (size_t)PACKED_DATA_ELEMENT_INITIAL;
                    height_element = (size_t)HEIGHT_ELEMENT_INITIAL;
                }

                data_index = (size_t)local_z * ((size_t)requester_element_x * (size_t)requester_element_y) + (size_t)local_y * (size_t)requester_element_x + (size_t)local_x;
                height_element++;
                data_index_byte = (size_t)data_index * (size_t)unitsize;
                data_onepack_byte = (size_t)data_onepack_element * (size_t)unitsize;
                if (target_local_rank_pre == INITIAL)
                {
                    target_local_rank_pre = target_local_rank_current;
                    real_buffer_offset = to_offset_from_coordinate(global_x, global_y, global_z); // + (size_t)buffer_index * (size_t)real_buffer_oneslot_allprocs_element;
                    req_packing_index = data_index_byte;
                    packed_data_element += (size_t)data_onepack_element;
                    packed_data_byte += (size_t)data_onepack_element * (size_t)unitsize;
                    if (packed_data_byte >= MAX_PACKING_INDEX)
                    {
                        fprintf(stderr, "Exceeded MAX_PACKING_INDEX in initialize_requester_write_info,myrank=%d,requester_write_times=%zu, packed_data_byte=%zu,max=%zu\n", submyrank, requester_write_times, packed_data_byte, MAX_PACKING_INDEX);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    continue;
                }
            }
        }

        CTCAR_get_grank_wrk(worker_wgid, target_local_rank_pre, &target_world_rank);

        // 最後のデータを格納

        requester_write_info_list[requester_write_times] = (requester_write_info_t){
            .src_data_offset = req_packing_index,
            .width_element = width_element,
            .height_element = height_element,
            .dest_data_offset = real_buffer_offset,
            .target_rank = target_world_rank};
        target_local_rank_pre = INITIAL;
        packed_data_element = (size_t)PACKED_DATA_ELEMENT_INITIAL;
        packed_data_byte = (size_t)PACKED_DATA_ELEMENT_INITIAL;
        height_element = (size_t)HEIGHT_ELEMENT_INITIAL;
        requester_write_times++;
        if (requester_write_times >= MAX_COMM_TIMES)
        {
            fprintf(stderr, "Exceeded MAX_COMM_TIMES in initialize_requester_write_info\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}
void initialize_con_packing_count()
{
    if (requester_write_times <= CON_PACKING_COUNT_MAX)
    {
        con_packing_count = requester_write_times;
    }
    else if (requester_write_times % CON_PACKING_COUNT_MAX == 0)
    {
        con_packing_count = CON_PACKING_COUNT_MAX;
    }
    else
    {
        printf("requester_write_times%d divided con_packing_count%d is not 0, so abort\n", requester_write_times, CON_PACKING_COUNT_MAX);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    loop_packing_write_count = requester_write_times / con_packing_count;
}

void reshape_requester_write_info()
{
#ifdef _ENABLE_CUDA_

    err = cudaMalloc((void **)&write_info_D, 3 * requester_write_times * sizeof(size_t));
    if (err != cudaSuccess)
    {
        printf("CUDA error:  cudaMalloc write_info_D: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t *height_element_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));
    size_t *width_element_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));
    size_t *width_byte_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));
    size_t *packed_data_element_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));
    size_t *packed_data_byte_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));
    size_t *src_data_offset_tmp = (size_t *)malloc(con_packing_count * sizeof(size_t));

    packed_data_byte_D = write_info_D;
    src_data_offset_D = write_info_D + requester_write_times;
    width_byte_D = write_info_D + 2 * requester_write_times;
    loop_ctr_max = requester_write_times / con_packing_count;
    size_t con_packing_byte = con_packing_count * sizeof(size_t);
    for (int loop_ctr = 0; loop_ctr < loop_ctr_max; loop_ctr++)
    {
        for (int i = 0; i < con_packing_count; i++)
        {
            height_element_tmp[i] = requester_write_info_list[loop_ctr * con_packing_count + i].height_element;
            width_element_tmp[i] = requester_write_info_list[loop_ctr * con_packing_count + i].width_element;
            width_byte_tmp[i] = width_element_tmp[i] * unitsize;
            packed_data_element_tmp[i] = height_element_tmp[i] * width_element_tmp[i];
            packed_data_byte_tmp[i] = packed_data_element_tmp[i] * unitsize;
            src_data_offset_tmp[i] = requester_write_info_list[loop_ctr * con_packing_count + i].src_data_offset;
            if (packed_data_byte_tmp[i] > packed_data_byte_max)
                packed_data_byte_max = packed_data_byte_tmp[i];
        }
        cudaMemcpy(&packed_data_byte_D[loop_ctr * con_packing_count], packed_data_byte_tmp, con_packing_byte, cudaMemcpyHostToDevice);
        cudaMemcpy(&src_data_offset_D[loop_ctr * con_packing_count], src_data_offset_tmp, con_packing_byte, cudaMemcpyHostToDevice);
        cudaMemcpy(&width_byte_D[loop_ctr * con_packing_count], width_byte_tmp, con_packing_byte, cudaMemcpyHostToDevice);
    }
    // debug
    //  size_t *write_info_D_debug = malloc(3 * requester_write_times * sizeof(size_t));
    //  cudaMemcpy(write_info_D_debug, write_info_D, 3 * requester_write_times * sizeof(size_t), cudaMemcpyDeviceToHost);
    //  for (int i = 0; i < 3 * requester_write_times; i++)
    //  {
    //      printf("rank=%d, write_info_D_debug[%d]=%zu\n", submyrank, i, write_info_D_debug[i]);
    //  }
    //  MPI_Barrier(CTCA_subcomm);
    // free(write_info_D_debug);
    free(height_element_tmp);
    free(width_element_tmp);
    free(width_byte_tmp);
    free(packed_data_element_tmp);
    free(packed_data_byte_tmp);
    free(src_data_offset_tmp);

#else
#endif
}

void show_header_buffer()
{
    for (int i = 0; i < MAX_BUFFER_SLOT; i++)
    {
        printf("myrank(l,w)=(%d,%d): [%d]=%d\n", submyrank, world_myrank, i, header_buffer[i]);
    }
}

void swap_data_addr(void **data0, void **data1)
{
    void *tmp;
    tmp = *data0;
    *data0 = *data1;
    *data1 = tmp;
}

void swap_data_real8(size_t **data0, size_t **data1)
{
    size_t *tmp;
    tmp = *data0;
    *data0 = *data1;
    *data1 = tmp;
}
void swap_data_int(int **data0, int **data1)
{
    int *tmp;
    tmp = *data0;
    *data0 = *data1;
    *data1 = tmp;
}

float buffer_output_float_time(struct timespec start_time, struct timespec end_time)
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

void buffer_calc_measure_time(float *elapsed_time, int iteration, float *average_time, float *max_time, float *min_time)
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
float buffer_reduce_max_gtime(float *ltime_list, float *max_gtime, int step_num, MPI_Comm new_comm)
{
    for (int i = 0; i < step_num; i++)
    {
        MPI_Reduce(&ltime_list[i], &max_gtime[i], 1, MPI_FLOAT, MPI_MAX, 0, new_comm);
    }
}

void requester_print_detail_write_time(int skip, int loop)
{
    float *write_polling_max_ranktime = (float *)malloc(sizeof(float) * write_time_ctr);
    if (write_polling_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: write_polling_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *write_max_ranktime = (float *)malloc(sizeof(float) * write_time_ctr);
    if (write_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: write_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *packing_max_ranktime = (float *)malloc(sizeof(float) * packing_time_ctr);
    if (packing_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: packing_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *all_garea_write_max_ranktime = (float *)malloc(sizeof(float) * all_garea_write_time_ctr);
    if (all_garea_write_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: all_garea_write_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int measure_skip = skip * loop_packing_write_count;
    int measure_loop = loop * loop_packing_write_count;
    buffer_reduce_max_gtime(&write_polling_time_list[skip], write_polling_max_ranktime, write_time_ctr - skip, CTCA_subcomm);
    buffer_reduce_max_gtime(&write_time_list[skip], write_max_ranktime, write_time_ctr - skip, CTCA_subcomm);
    buffer_reduce_max_gtime(&packing_time_list[measure_skip], packing_max_ranktime, measure_loop, CTCA_subcomm);
    buffer_reduce_max_gtime(&all_garea_write_time_list[measure_skip], all_garea_write_max_ranktime, measure_loop, CTCA_subcomm);
    if (submyrank == 0)
    {
        // ステップごとのpolling_timeとwrite_timeの統計表示
        //  for (int i = 0; i < write_time_ctr; i++)
        //  {
        //      printf("requester_write_info[%d]: write_polling_time=%f sec, write_time=%f sec\n", i, write_polling_max_ranktime[i], write_max_ranktime[i]);
        //  }
        float write_polling_max_gtime, write_polling_min_gtime, write_polling_ave_gtime;
        float write_max_gtime, write_min_gtime, write_ave_gtime;
        float packing_max_gtime, packing_min_gtime, packing_ave_gtime;
        float all_garea_write_max_gtime, all_garea_write_min_gtime, all_garea_write_ave_gtime;

        buffer_calc_measure_time(write_polling_max_ranktime, write_time_ctr - skip, &write_polling_ave_gtime, &write_polling_max_gtime, &write_polling_min_gtime);
        buffer_calc_measure_time(write_max_ranktime, write_time_ctr - skip, &write_ave_gtime, &write_max_gtime, &write_min_gtime);
        buffer_calc_measure_time(packing_time_list, measure_loop, &packing_ave_gtime, &packing_max_gtime, &packing_min_gtime);
        buffer_calc_measure_time(all_garea_write_time_list, measure_loop, &all_garea_write_ave_gtime, &all_garea_write_max_gtime, &all_garea_write_min_gtime);
        printf("skip=%d, loop=%d, packing_time_ctr=%d, all_garea_write_time_ctr=%d, loop_packing_write_count=%d\n", skip, loop, packing_time_ctr, all_garea_write_time_ctr, loop_packing_write_count);
        printf("req write_polling_time: ave=%f, max=%f, min=%f sec\n", write_polling_ave_gtime, write_polling_max_gtime, write_polling_min_gtime);
        printf("req write_realwrite_time: ave=%f, max=%f, min=%f sec\n", write_ave_gtime, write_max_gtime, write_min_gtime);
        printf("req packing_time: ave=%f, max=%f, min=%f sec\n", packing_ave_gtime * loop_packing_write_count, packing_max_gtime * loop_packing_write_count, packing_min_gtime * loop_packing_write_count);
        printf("req all_garea_write_time: ave=%f, max=%f, min=%f sec\n", all_garea_write_ave_gtime * loop_packing_write_count, all_garea_write_max_gtime * loop_packing_write_count, all_garea_write_min_gtime * loop_packing_write_count);
    }
    free(write_polling_max_ranktime);
    free(write_max_ranktime);
    free(packing_max_ranktime);
    free(all_garea_write_max_ranktime);
}

void worker_print_detail_read_time()
{
    float *read_polling_max_ranktime = (float *)malloc(sizeof(float) * read_time_ctr);
    if (read_polling_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: read_polling_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *read_max_ranktime = (float *)malloc(sizeof(float) * read_time_ctr);
    if (read_max_ranktime == NULL)
    {
        fprintf(stderr, "malloc error: read_max_ranktime\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 全てのランクが全てのステップのpolling_timeとread_timeを集計する
    //  for (int i = 0; i < read_time_ctr; i++)
    //  {
    //      printf("myrank=%d,worker_read_info[%d]: read_polling_time=%f, read_time=%f sec\n", submyrank, i, read_polling_time_list[i], read_time_list[i]);
    //  }

    buffer_reduce_max_gtime(read_polling_time_list, read_polling_max_ranktime, read_time_ctr, CTCA_subcomm);
    buffer_reduce_max_gtime(read_time_list, read_max_ranktime, read_time_ctr, CTCA_subcomm);
    if (submyrank == 0)
    {
        // ステップごとのpolling_timeとread_timeの統計表示
        // for (int i = 0; i < read_time_ctr; i++)
        // {
        //     printf("worker_read_info[%d]: read_polling_time=%f, read_time=%f sec\n", i, read_polling_max_ranktime[i], read_max_ranktime[i]);
        // }
        float read_polling_max_gtime, read_polling_min_gtime, read_polling_ave_gtime;
        float read_max_gtime, read_min_gtime, read_ave_gtime;

        buffer_calc_measure_time(read_polling_max_ranktime, read_time_ctr, &read_polling_ave_gtime, &read_polling_max_gtime, &read_polling_min_gtime);
        buffer_calc_measure_time(read_max_ranktime, read_time_ctr, &read_ave_gtime, &read_max_gtime, &read_min_gtime);
        printf("wrk read_polling_time: ave=%f, max=%f, min=%f sec\n", read_polling_ave_gtime, read_polling_max_gtime, read_polling_min_gtime);
        printf("wrk read_realdata_time: ave=%f, max=%f, min=%f sec\n", read_ave_gtime, read_max_gtime, read_min_gtime);
    }
    free(read_polling_max_ranktime);
    free(read_max_ranktime);
}
void count_buffer_index()
{
    if (submyrank == 0)
    {
        size_t height_element = requester_write_info_list[0].height_element;
        size_t width_element = requester_write_info_list[0].width_element;
        size_t width_byte = width_element * unitsize;
        size_t packed_data_element0 = height_element * width_element;
        size_t packed_data_byte = packed_data_element0 * unitsize;
        size_t con_packing_count;
        if (requester_write_times <= CON_PACKING_COUNT_MAX)
        {
            con_packing_count = requester_write_times;
        }
        else if (requester_write_times % CON_PACKING_COUNT_MAX == 0)
        {
            con_packing_count = CON_PACKING_COUNT_MAX;
        }
        printf("packed_data_byte=%zu\n", packed_data_byte);
        printf("con_packing_count=%zu\n", con_packing_count);
        printf("buffer_index[0]=%d,[1]=%d,[2]=%d,[3]=%d\n", buffer_index_counter[0], buffer_index_counter[1], buffer_index_counter[2], buffer_index_counter[3]);
        printf("requester_write_times=%d\n", requester_write_times);
    }
}
void initialize_count_buffer_index()
{
    for (int i = 0; i < MAX_BUFFER_SLOT; i++)
    {
        buffer_index_counter[i] = 0;
    }
}