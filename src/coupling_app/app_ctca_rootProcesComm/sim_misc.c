#include "sim_misc.h"
#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "kernel.h"

static time_t sec_org = 0;
static time_t usec_org = 0;

double get_elapsed_sec_time(const struct timeval *tv0, const struct timeval *tv1)
{
  return (double)(tv1->tv_sec - tv0->tv_sec) + (double)(tv1->tv_usec - tv0->tv_usec) * 1.0e-6;
}
double get_elapsed_usec_time(const struct timeval *tv0, const struct timeval *tv1)
{
  return (double)(tv1->tv_sec - tv0->tv_sec) * 1.0e+6 + (double)(tv1->tv_usec - tv0->tv_usec);
}
int omb_get_local_rank()
{
  char *str = NULL;
  int local_rank = -1;

  if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL)
  {
    local_rank = atoi(str);
  }
  else if ((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL)
  {
    local_rank = atoi(str);
  }
  else if ((str = getenv("MPI_LOCALRANKID")) != NULL)
  {
    local_rank = atoi(str);
  }
  else if ((str = getenv("LOCAL_RANK")) != NULL)
  {
    local_rank = atoi(str);
  }
  else
  {
    fprintf(stderr, "Warning: OMB could not identify the local rank of the process.\n");
    fprintf(stderr, "         This can lead to multiple processes using the same GPU.\n");
    fprintf(stderr, "         Please use the get_local_rank script in the OMB repo for this.\n");
  }

  return local_rank;
}

void print_now_time()
{
  time_t now;
  time(&now);

  // 現在の時刻をローカルタイムに変換
  struct tm *local = localtime(&now);

  // フォーマットして出力
  printf("現在の時刻: %04d-%02d-%02d %02d:%02d:%02d\n",
         local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
         local->tm_hour, local->tm_min, local->tm_sec);
}

void swap(float **f, float **fn)
{
  float *tmp;
  tmp = *f;
  *f = *fn;
  *fn = tmp;
}

void start_timer()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);

  sec_org = tv.tv_sec;
  usec_org = tv.tv_usec;
}

double get_elapsed_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return (double)(tv.tv_sec - sec_org) + (double)(tv.tv_usec - usec_org) * 1.0e-6;
}
// visプログラムへプロセス計算を行い、fの値を送る
int comm_data(float *tmp, float *f, float *result, int nx, int ny, int nz, int nx0, int nz0, int new_rank, int rank, int new_nprocs, int nprocs, int num, int calc, MPI_Comm new_comm, int Px, int Pz)
{
  const int tag = 0;
  int to_proc = 0;
  int lenx = num / Px;
  int lenz = num / Pz;
  int leng = num / new_nprocs;
  float buf[128];
  int i, j, k, y;
  for (y = 0; y < ny; y++)
  {
    // printf("Sim_PD 0 : new_rank=%d, y=%d\n",new_rank,y);
    for (i = 0; i < leng; i++)
    {
      // どのvisプロセスに送信するかを決定
      for (j = 1; j <= Pz; j++)
      {
        int sim_column_num = i + new_rank * leng;
        if (sim_column_num < lenz * j)
        {
          to_proc = nprocs - Px * j;
          break;
        }
      }
      for (j = 0; j < Px; j++)
      {
        // visプロセスへデータを送信
        int cell = nx * ny * nz + y * nx - i * nx * ny + j * lenx;
        MPI_Send(&f[nx * ny * nz + y * nx - i * nx * ny + j * lenx], lenx, MPI_FLOAT, to_proc, tag, MPI_COMM_WORLD);
        // どのvisプロセスに送信するかを決定
        to_proc++;
      }
    }
  }
  return 0;
}
// debug用にファイルへ出力する
void print_slice_csvfile(float *reverse_slice_f, float *f_H, float *result, int nx, int ny, int nz, int nx0, int nz0, int new_rank, int new_nprocs, int num, int step, MPI_Comm new_comm)
{
  const int tag = 0;
  int i, j, k, y;
  MPI_Status stat[32];
  for (y = 0; y < ny; y++)
  {
    // printf("fから読み込み: y=%d,calc=%d\n",y,calc);
    for (i = 0; i < nz; i++)
    {
      for (j = 0; j < nx; j++)
      {
        // printf("fから読み込み: y=%d,calc=%d,i=%d,j=%d\n",y,calc,i,j);
        reverse_slice_f[i * nx + j] = f_H[nx * y + nx * ny * (nz - i) + j];
      }
    }
    gather_y_axis_normal_plane_H(reverse_slice_f, result, nx, nz, new_rank, new_nprocs, new_comm);
    if (new_rank == 0)
    {
      char directory_name[100];
      char file_name[100];
      sprintf(directory_name, "test%d_step%d", nx, step);
      sprintf(file_name, "y_%d", y);
      print_csvfile(result, num, num, directory_name, file_name);
    }
  }
}

// y_axis_normal_plane_perprocsの値をプロセス0へ集める
void gather_y_axis_normal_plane_D(float *y_axis_normal_plane_perprocs, float *y_axis_normal_plane_allprocs, int nx, int nz, int new_rank, int new_nprocs, MPI_Comm new_comm, MPI_Request *req, cudaStream_t packing_stream)
{
  int i, j, k;
  int req_count = 0;
  int isend_tag = 0;
  if (new_rank == 0)
  {
    for (i = 1; i < new_nprocs; i++)
    {
      int source_rank = new_nprocs - i;
      int irecv_tag = 0;
      for (j = 0; j < nz; j++)
      {
        MPI_Irecv(&y_axis_normal_plane_allprocs[nx * nz * (i - 1) + nx * j], nx, MPI_FLOAT, source_rank, irecv_tag, new_comm, &req[req_count++]);
        irecv_tag++;
      }
    }
    call_packing_data(&y_axis_normal_plane_allprocs[nx * nz * (new_nprocs - 1)], y_axis_normal_plane_perprocs, nx, 0, nx, nx, nz, packing_stream);
  }
  else
  {
    for (i = 0; i < nz; i++)
    {
      int dest_rank = 0;
      MPI_Isend(&y_axis_normal_plane_perprocs[i * nx], nx, MPI_FLOAT, dest_rank, isend_tag, new_comm, &req[req_count++]);
      isend_tag++;
    }
  }
}

void gather_y_axis_normal_plane_H(float *y_axis_normal_plane_perprocs, float *y_axis_normal_plane_allprocs, int nx, int nz, int new_rank, int new_nprocs, MPI_Comm new_comm)
{
  const int tag = 0;
  int i, j, k;
  if (new_rank == 0)
  {
    for (i = 1; i < new_nprocs; i++)
    {
      int source_rank = new_nprocs - i;
      for (j = 0; j < nz; j++)
      {
        MPI_Recv(&y_axis_normal_plane_allprocs[nx * nz * (i - 1) + nx * j], nx, MPI_FLOAT, source_rank, tag, new_comm, MPI_STATUS_IGNORE);
      }
    }
    for (i = 0; i < nz; i++)
    {
      memcpy(&y_axis_normal_plane_allprocs[nx * nz * (new_nprocs - 1) + i * nx], &y_axis_normal_plane_perprocs[i * nx], nx * sizeof(float));
    }
  }
  else
  {
    for (i = 0; i < nz; i++)
    {
      int dest_rank = 0;
      MPI_Send(&y_axis_normal_plane_perprocs[i * nx], nx, MPI_FLOAT, dest_rank, tag, new_comm);
    }
  }
}

void print_y_axis_normal_plane(float *y_axis_normal_plane_allprocs, int step, int y, int num, int nx0, int nz0, char *file_head_name)
{
  char directory_name[100];
  char file_name[100];
  sprintf(directory_name, "test%d_step%d", num, step);
  sprintf(file_name, "%s_y%d", file_head_name, y);
  print_csvfile(y_axis_normal_plane_allprocs, nx0, nz0, directory_name, file_name);
}

void print_y_axis_normal_plane_perprocs(float *y_axis_normal_plane_perprocs, int step, int y, int num, int nx, int nz, int new_rank, char *file_head_name)
{
  char directory_name[100];
  char file_name[100];
  sprintf(directory_name, "test%d_step%d", num, step);
  sprintf(file_name, "%s_y%d_newrank%d", file_head_name, y, new_rank);
  print_csvfile(y_axis_normal_plane_perprocs, nx, nz, directory_name, file_name);
}

void print_csvfile(float *result, int x_num, int y_num, char *directory_name, char *file_name)
{
  FILE *fp;
  char file_pass[100];
  int i, j;
  sprintf(file_pass, "./%s/%s.csv", directory_name, file_name);
  fp = fopen(file_pass, "w");
  for (i = 0; i < y_num; i++)
  {
    for (j = 0; j < x_num; j++)
    {
      if (j != 0)
        fprintf(fp, ",");
      fprintf(fp, "%.3f", result[i * x_num + j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}
