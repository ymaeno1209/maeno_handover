#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "png.h"

#define SIGNATURE_NUM 8

typedef struct
{
    unsigned char *data;
    unsigned int width;
    unsigned int height;
    unsigned int ch;
} BITMAPDATA_t;
void print_y_axis_normal_plane(float *y_axis_normal_plane_allprocs, int step, int y, int num, char *file_head_name);
void rgb_comm(int new_rank, int new_nprocs, int Px, int lenx, int lenz, int x, int y, int a, int b, BITMAPDATA_t bitmap, MPI_Comm new_comm);
void recv_data(float *print_buf, int new_rank, int lenx, int lenz, int leng, int Px, int Pg, int b);
void drawDot(unsigned char *data, unsigned int width, unsigned int height, unsigned int x1, unsigned int y1, float value);
double get_elapsed_sec_time(const struct timeval *tv0, const struct timeval *tv1);
double get_elapsed_usec_time(const struct timeval *tv0, const struct timeval *tv1);
int pngFileReadDecode(BITMAPDATA_t *, const char *);
int pngFileEncodeWrite(BITMAPDATA_t *, const char *);
int freeBitmapData(BITMAPDATA_t *);
void rgb_comm_to_allprocs(int num, int new_rank, int new_nprocs, int x_division, int length_x, int length_z, int vis_x, int vis_y, int rank_Px, int rank_Pz, BITMAPDATA_t *src_bitmap, BITMAPDATA_t *dest_bitmap, MPI_Comm new_comm, MPI_Request *req);
