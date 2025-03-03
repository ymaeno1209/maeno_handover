#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "ctca.h"

typedef enum
{
    XY,
    YZ,
    ZX
} target_side_type;

typedef struct requester_write_info_t
{
    size_t src_data_offset;
    size_t width_element;
    size_t height_element;
    size_t dest_data_offset;
    int target_rank;
} requester_write_info_t;

void requester_buffer_init(int *requester_data_division, size_t datasize, bool data_direction, int type);

void requester_buffer_init_withint(int *requester_data_division, size_t datasize, bool data_direction);
void requester_buffer_init_withreal4(int *requester_data_division, size_t datasize, bool data_direction);
void requester_buffer_init_withreal8(int *requester_data_division, size_t datasize, bool data_direction);

void requester_buffer_write(void *src_data, int step);
void requester_buffer_fin();

void worker_buffer_init(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length, int type);
void worker_buffer_init_withint(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);
void worker_buffer_init_withreal4(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);
void worker_buffer_init_withreal8(int *worker_data_division, target_side_type target_side, int *worker_data_oneside_length);

void worker_buffer_read(void *dest_data, int step);
void worker_buffer_fin();

void coupler_buffer_init();
void coupler_buffer_fin();

int *worker_collect_init_info(int *worker_data_division, target_side_type target_side);

MPI_Datatype decide_MPI_datatype(int type);
int decide_unitsize(int type);

void garea_write_autotype(int real_buffer_gareaid, int *target_world_rank, size_t *offset, size_t *packed_data_element, void *comm_data, int con_packing_count);
void fetch_worker_init_info(int *worker_init_info);
void fetch_requester_init_info(int *requester_init_info);
void setup_conditions(int *worker_init_info, int *requester_data_division);
int to_worker_localrank_from_coordinate(int global_x, int global_y, int global_z);
size_t to_offset_from_coordinate(int global_x, int global_y, int global_z);
void to_global_coordinate_from_local_coordinate_x(int local_x, int *global_x);
void to_global_coordinate_from_local_coordinate_y(int local_y, int *global_y);
void to_global_coordinate_from_local_coordinate_z(int local_z, int *global_z);
int requester_polling_header_buffer();
int worker_polling_header_buffer(int step);
void write_header_buffer(int buffer_index, int step);
void requester_write_onestep_to_realbuffer(void *src_data, int buffer_index);
void initialize_header_buffer();
void initialize_real_buffer();
void clear_head_buffer_at_index(int buffer_index);
void initialize_requester_write_info();
void initialize_con_packing_count();
void reshape_requester_write_info();
void swap_data_addr(void **data0, void **data1);
void swap_data_real8(size_t **data0, size_t **data1);
void swap_data_int(int **data0, int **data1);

// function for debug
void show_header_buffer();
void initialize_count_buffer_index();
void count_buffer_index();
void requester_print_detail_write_time(int skip, int loop);
void worker_print_detail_read_time();