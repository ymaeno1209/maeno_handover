#include <mpi.h>
#include <sys/time.h>
#include "cuda_runtime.h"

void print_y_axis_normal_plane_perprocs(float *y_axis_normal_plane_perprocs, int step, int y, int num, int nx, int nz, int new_rank, char *file_head_name);
void print_y_axis_normal_plane(float *y_axis_normal_plane_allprocs, int step, int y, int num, int nx0, int nz0, char *file_head_name);
void gather_y_axis_normal_plane_D(float *y_axis_normal_plane_perprocs, float *y_axis_normal_plane_allprocs, int nx, int nz, int new_rank, int new_nprocs, MPI_Comm new_comm, MPI_Request *req, cudaStream_t packing_stream);
void gather_y_axis_normal_plane_H(float *y_axis_normal_plane_perprocs, float *y_axis_normal_plane_allprocs, int nx, int nz, int new_rank, int new_nprocs, MPI_Comm new_comm);
double get_elapsed_sec_time(const struct timeval *tv0, const struct timeval *tv1);
double get_elapsed_usec_time(const struct timeval *tv0, const struct timeval *tv1);
int omb_get_local_rank();
void print_now_time();
void swap(float **f, float **fn);
void start_timer();
double get_elapsed_time();
int comm_data(float *tmp, float *f, float *result, int nx, int ny, int nz, int nx0, int nz0, int new_rank, int rank, int new_nprocs, int nprocs, int num, int calc, MPI_Comm new_comm, int Px, int Pz);
void print_slice_csvfile(float *reverse_slice_f, float *f_H, float *result, int nx, int ny, int nz, int nx0, int nz0, int new_rank, int new_nprocs, int num, int step, MPI_Comm new_comm);
void print_csvfile(float *result, int x_num, int y_num, char *directory_name, char *file_name);
