#include "myPng.h"
#include <sys/time.h>
#include "mpi.h"
#include <math.h> // sqrt 関数を使用

void print_y_axis_normal_plane(float *y_axis_normal_plane_allprocs, int step, int y, int num, char *file_head_name)
{
  int i, j;
  FILE *fp;
  char filename[100];
  sprintf(filename, "./test%d_step%d/%sy_%d.csv", num, step, file_head_name, y);
  fp = fopen(filename, "w");
  for (i = 0; i < num; i++)
  {
    for (j = 0; j < num; j++)
    {
      if (j != 0)
        fprintf(fp, ",");
      fprintf(fp, "%.3f", y_axis_normal_plane_allprocs[i * num + j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void rgb_comm(int new_rank, int new_nprocs, int Px, int lenx, int lenz, int x, int y, int a, int b, BITMAPDATA_t bitmap, MPI_Comm new_comm)
{
  int i, j;
  int tag = 0;
  unsigned char *start_pos = bitmap.data + y * bitmap.width * 3 + x * 3;
  unsigned char *procs_start_pos = start_pos + lenx * a * 3 + lenz * bitmap.width * b * 3;
  if (new_rank == 0)
  {
    for (i = 1; i < new_nprocs; i++)
    {
      int aa = i % Px;
      int bb = i / Px;
      unsigned char *rank_start_pos = start_pos + lenx * aa * 3 + lenz * bitmap.width * bb * 3;

      for (j = 0; j < lenz; j++)
      {
        MPI_Recv(rank_start_pos + j * bitmap.width * 3, lenx * 3, MPI_CHAR, i, tag, new_comm, MPI_STATUS_IGNORE);
      }
    }
  }
  else
  {
    for (i = 0; i < lenz; i++)
    {
      MPI_Send(procs_start_pos + i * bitmap.width * 3, lenx * 3, MPI_CHAR, 0, tag, new_comm);
    }
  }
}
void rgb_comm_to_allprocs(int num, int new_rank, int new_nprocs, int x_division, int length_x, int length_z, int vis_x, int vis_y, int rank_Px, int rank_Pz, BITMAPDATA_t *src_bitmap, BITMAPDATA_t *dest_bitmap, MPI_Comm new_comm, MPI_Request *req)
{
  int req_count = 0;
  int every_procs_print_y_element = num / new_nprocs;
  int print_bitmap_width = dest_bitmap[0].width;
  size_t comm_size = (size_t)length_x * (size_t)3;
  size_t src_bitmap_index, src_bitmap_data_index, dest_bitmap_index, dest_bitmap_data_index;
  size_t src_bitmap_data_index_max = (size_t)src_bitmap[0].height * (size_t)src_bitmap[0].width * (size_t)src_bitmap[0].ch;
  size_t dest_bitmap_data_index_max = (size_t)dest_bitmap[0].height * (size_t)dest_bitmap[0].width * (size_t)dest_bitmap[0].ch;
  for (int i = 0; i < every_procs_print_y_element; i++)
  {
    for (int j = 0; j < length_z; j++)
    {
      src_bitmap_index = i + new_rank * every_procs_print_y_element;
      src_bitmap_data_index = 3L * j * length_x;
      dest_bitmap_index = i;
      dest_bitmap_data_index = 3L * (vis_y * print_bitmap_width + vis_x + rank_Px * length_x + rank_Pz * print_bitmap_width * length_z + j * print_bitmap_width);
      // printf("memcpy(%d,%d): myrank=%d, src_bitmap_index=%zu, src_bitmap_data_index=%zu, dest_bitmap_index=%zu, dest_bitmap_data_index=%zu\n", i, j, new_rank, src_bitmap_index, src_bitmap_data_index, dest_bitmap_index, dest_bitmap_data_index);
      memcpy(&dest_bitmap[dest_bitmap_index].data[dest_bitmap_data_index], &src_bitmap[src_bitmap_index].data[src_bitmap_data_index], comm_size);
    }
  }
  int irecv_tag = 0;
  for (int i = 0; i < num; i++)
  {
    if (i / every_procs_print_y_element == new_rank)
    {
      continue;
    }
    if (i % every_procs_print_y_element == 0)
    {
      irecv_tag = 0;
    }

    int source_rank = i / every_procs_print_y_element;
    int source_rank_Px = source_rank % x_division;
    int source_rank_Pz = source_rank / x_division;
    int dest_bitmap_index_count = 0;
    dest_bitmap_index = i % every_procs_print_y_element;
    for (int j = 0; j < length_z; j++)
    {

      dest_bitmap_data_index = 3L * (vis_y * print_bitmap_width + vis_x + source_rank_Px * length_x + source_rank_Pz * print_bitmap_width * length_z + j * print_bitmap_width);

      if (dest_bitmap_data_index > dest_bitmap_data_index_max)
      {
        printf("rgb_comm_irecv(%d,%d): error: dest_bitmap_data_index=%zu, dest_bitmap_data_index_max=%zu\n", i, j, dest_bitmap_data_index, dest_bitmap_data_index_max);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      // printf("rgb_comm_irecv(%d,%d): myrank=%d,source_rank=%d, dest_bitmap_index=%d,dest_bitmap_data_index=%zu,comm_size=%zu,irecv_tag=%d\n", i, j, new_rank, source_rank, dest_bitmap_index, dest_bitmap_data_index, comm_size, irecv_tag);
      MPI_Irecv(&dest_bitmap[dest_bitmap_index].data[dest_bitmap_data_index], comm_size, MPI_CHAR, source_rank, irecv_tag, new_comm, &req[req_count++]);
      irecv_tag++;
    }
  }

  int isend_tag = 0;
  int dest_rank_ctr = -1;

  for (int i = 0; i < num; i++)
  {
    if (i % every_procs_print_y_element == 0)
    {
      dest_rank_ctr++;
    }
    if (new_rank * every_procs_print_y_element <= i && i < (new_rank + 1) * every_procs_print_y_element)
    {
      continue;
    }

    int dest_rank = dest_rank_ctr;
    src_bitmap_index = i;
    if (i % every_procs_print_y_element == 0)
    {
      isend_tag = 0;
    }

    for (int j = 0; j < length_z; j++)
    {
      src_bitmap_data_index = 3L * (size_t)j * (size_t)length_x;
      if (src_bitmap_data_index > src_bitmap_data_index_max)
      {
        printf("rgb_comm_isend(%d,%d): error: src_bitmap_data_index=%zu, src_bitmap_data_index_max=%zu\n", i, j, src_bitmap_data_index, src_bitmap_data_index_max);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      // printf("rgb_comm_isend(%d,%d): myrank=%d,dest_rank=%d,src_bitmap_index=%zu,src_bitmap_data_index=%zu, comm_size=%zu, isend_tag=%d\n", i, j, new_rank, dest_rank, src_bitmap_index, src_bitmap_data_index, comm_size, isend_tag);
      MPI_Isend(&src_bitmap[src_bitmap_index].data[src_bitmap_data_index], comm_size, MPI_CHAR, dest_rank, isend_tag, new_comm, &req[req_count++]);
      isend_tag++;
    }
  }
}

void recv_data(float *print_buf, int new_rank, int lenx, int lenz, int leng, int Px, int Pg, int b)
{
  int tag = 0;
  int from_rank = -1;
  int offset_cnt = -1;
  int from_rank_pre = -1;
  int same_cnt = 0;
  for (int i = 0; i < lenz; i++)
  {
    for (int j = 0; j < Pg; j++)
    {
      if (i + lenz * b < leng * (j + 1))
      {
        from_rank_pre = Pg - 1 - j;
        break;
      }
    }
    if (from_rank != from_rank_pre)
    {
      from_rank = from_rank_pre;
      offset_cnt++;
      same_cnt = 0;
    }
    else
      same_cnt++;
    MPI_Recv(&print_buf[same_cnt * lenx + lenx * offset_cnt * leng], lenx, MPI_FLOAT, from_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}
void drawDot(
    unsigned char *data, /* ビットマップデータ */
    unsigned int width,  /* ビットマップの横幅 */
    unsigned int height, /* ビットマップの高さ */
    unsigned int x1,     /* 始点のx座標 */
    unsigned int y1,     /* 始点のy座標 */
    float value          /* 値 (0.0 ～ 1.0) */
)
{
  unsigned char *p;
  p = data + y1 * width * 3 + x1 * 3;

  // 値を0.0～1.0にクランプ
  if (value < 0.0)
    value = 0.0;
  if (value > 1.0)
    value = 1.0;

  // 非線形スケーリング (0に近いほど変化を細かく)
  float scaledValue = sqrt(value); // √valueを使うと0に近い変化が細かくなる

  // グラデーションの計算
  unsigned char red = (unsigned char)(255 * scaledValue);          // 赤はスケール値に比例
  unsigned char blue = (unsigned char)(255 * (1.0 - scaledValue)); // 青は1 - スケール値に比例
  unsigned char green = 0;                                         // 緑は使わない

  // 色を設定
  p[0] = red;   // 赤
  p[1] = green; // 緑
  p[2] = blue;  // 青
}
double get_elapsed_sec_time(const struct timeval *tv0, const struct timeval *tv1)
{
  return (double)(tv1->tv_sec - tv0->tv_sec) + (double)(tv1->tv_usec - tv0->tv_usec) * 1.0e-6;
}
double get_elapsed_usec_time(const struct timeval *tv0, const struct timeval *tv1)
{
  return (double)(tv1->tv_sec - tv0->tv_sec) * 1.0e+6 + (double)(tv1->tv_usec - tv0->tv_usec);
}

int pngFileReadDecode(BITMAPDATA_t *bitmapData, const char *filename)
{

  FILE *fi;
  int j;
  unsigned int width, height;
  unsigned int readSize;

  png_structp png;
  png_infop info;
  png_bytepp datap;
  png_byte type;
  png_byte signature[8];

  fi = fopen(filename, "rb");
  if (fi == NULL)
  {
    printf("%sは開けません\n", filename);
    return -1;
  }

  readSize = fread(signature, 1, SIGNATURE_NUM, fi);

  if (png_sig_cmp(signature, 0, SIGNATURE_NUM))
  {
    printf("png_sig_cmp error!\n");
    fclose(fi);
    return -1;
  }

  png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png == NULL)
  {
    printf("png_create_read_struct error!\n");
    fclose(fi);
    return -1;
  }

  info = png_create_info_struct(png);
  if (info == NULL)
  {
    printf("png_crete_info_struct error!\n");
    png_destroy_read_struct(&png, NULL, NULL);
    fclose(fi);
    return -1;
  }

  png_init_io(png, fi);
  png_set_sig_bytes(png, readSize);
  png_read_png(png, info, PNG_TRANSFORM_PACKING | PNG_TRANSFORM_STRIP_16, NULL);

  width = png_get_image_width(png, info);
  height = png_get_image_height(png, info);

  datap = png_get_rows(png, info);

  type = png_get_color_type(png, info);
  /* とりあえずRGB or RGBAだけ対応 */
  if (type != PNG_COLOR_TYPE_RGB && type != PNG_COLOR_TYPE_RGB_ALPHA)
  {
    printf("color type is not RGB or RGBA\n");
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fi);
    return -1;
  }

  bitmapData->width = width;
  bitmapData->height = height;
  if (type == PNG_COLOR_TYPE_RGB)
  {
    bitmapData->ch = 3;
  }
  else if (type == PNG_COLOR_TYPE_RGBA)
  {
    bitmapData->ch = 4;
  }
  printf("width = %d, height = %d, ch = %d\n", bitmapData->width, bitmapData->height, bitmapData->ch);

  bitmapData->data =
      (unsigned char *)malloc(sizeof(unsigned char) * bitmapData->width * bitmapData->height * bitmapData->ch);
  if (bitmapData->data == NULL)
  {
    printf("data malloc error\n");
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fi);
    return -1;
  }

  for (j = 0; j < bitmapData->height; j++)
  {
    memcpy(bitmapData->data + j * bitmapData->width * bitmapData->ch, datap[j], bitmapData->width * bitmapData->ch);
  }

  png_destroy_read_struct(&png, &info, NULL);
  fclose(fi);

  return 0;
}

int pngFileEncodeWrite(BITMAPDATA_t *bitmapData, const char *filename)
{
  FILE *fo;
  int j;

  png_structp png;
  png_infop info;
  png_bytepp datap;
  png_byte type;

  fo = fopen(filename, "wb");
  if (fo == NULL)
  {
    printf("%sは開けません\n", filename);
    return -1;
  }

  png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  info = png_create_info_struct(png);

  if (bitmapData->ch == 3)
  {
    type = PNG_COLOR_TYPE_RGB;
  }
  else if (bitmapData->ch == 4)
  {
    type = PNG_COLOR_TYPE_RGB_ALPHA;
  }
  else
  {
    printf("ch num is invalid!\n");
    png_destroy_write_struct(&png, &info);
    fclose(fo);
    return -1;
  }
  png_init_io(png, fo);
  png_set_IHDR(png, info, bitmapData->width, bitmapData->height, 8, type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  datap = png_malloc(png, sizeof(png_bytep) * bitmapData->height);

  png_set_rows(png, info, datap);

  for (j = 0; j < bitmapData->height; j++)
  {
    datap[j] = png_malloc(png, bitmapData->width * bitmapData->ch);
    memcpy(datap[j], bitmapData->data + j * bitmapData->width * bitmapData->ch, bitmapData->width * bitmapData->ch);
  }
  png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

  for (j = 0; j < bitmapData->height; j++)
  {
    png_free(png, datap[j]);
  }
  png_free(png, datap);

  png_destroy_write_struct(&png, &info);
  fclose(fo);
  return 0;
}

int freeBitmapData(BITMAPDATA_t *bitmap)
{
  if (bitmap->data != NULL)
  {
    free(bitmap->data);
    bitmap->data = NULL;
  }
  return 0;
}
