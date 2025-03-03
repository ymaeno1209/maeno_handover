#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>
#include "garea.h"

volatile static MPI_Aint *garea_base_addresses;
static int garea_base_addresses_byte;
static int world_myrank;
static int world_nprocs;
static int gareaidctr = 0;
static int garea_base_addresses_element_per_garea;
static MPI_Win *win_garea_table;
static MPI_Win win_garea_base_addresses;

#define MAX_TRANSFER_SIZE (1024 * 1024 * 1024)
#define GAREA_BASE_ADDRESSES_INITIAL -1
#define GAREA_BASE_ADDRESSES_ELEMENT_PER_WIN 1
#define DATA_INT 0
#define DATA_REAL4 1
#define DATA_REAL8 2
#define DEF_MAXNUM_GAREAS 10

void CTCAR_garea_create(int *gareaid)
{
    garea_create(DEF_MAXNUM_GAREAS, gareaid);
}

void CTCAC_garea_create(int *gareaid)
{
    garea_create(DEF_MAXNUM_GAREAS, gareaid);
}

void CTCAW_garea_create(int *gareaid)
{
    garea_create(DEF_MAXNUM_GAREAS, gareaid);
}

void CTCAR_garea_attach(int gareaid, void *base, int size_byte)
{
    garea_attach(gareaid, base, size_byte);
}

void CTCAC_garea_attach(int gareaid, void *base, int size_byte)
{
    garea_attach(gareaid, base, size_byte);
}

void CTCAW_garea_attach(int gareaid, void *base, int size_byte)
{
    garea_attach(gareaid, base, size_byte);
}

void CTCAR_garea_detach(int gareaid, void *base)
{
    garea_detach(gareaid, base);
}

void CTCAC_garea_detach(int gareaid, void *base)
{
    garea_detach(gareaid, base);
}

void CTCAW_garea_detach(int gareaid, void *base)
{
    garea_detach(gareaid, base);
}

void CTCAR_garea_delete()
{
    garea_delete();
}

void CTCAC_garea_delete()
{
    garea_delete();
}

void CTCAW_garea_delete()
{
    garea_delete();
}

void CTCAR_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_INT);
}

void CTCAR_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL4);
}

void CTCAR_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL8);
}

void CTCAC_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_INT);
}

void CTCAC_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL4);
}

void CTCAC_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL8);
}

void CTCAW_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_INT);
}

void CTCAW_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL4);
}

void CTCAW_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr)
{
    garea_read(gareaid, target_world_rank, offset, data_element, dest_addr, DATA_REAL8);
}

void CTCAR_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_INT);
}

void CTCAC_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_INT);
}

void CTCAW_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_INT);
}
void CTCAR_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL4);
}

void CTCAC_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL4);
}

void CTCAW_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL4);
}
void CTCAR_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL8);
}

void CTCAC_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL8);
}

void CTCAW_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr)
{
    garea_write(gareaid, target_world_rank, offset, data_element, src_addr, DATA_REAL8);
}
void garea_create(int num_gareas, int *gareaid)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_myrank);
    *gareaid = gareaidctr;

    if (gareaidctr == 0)
    {
        setup_common_tables(num_gareas);
    }

    initialize_garea(gareaidctr);

    gareaidctr++;
}

void garea_attach(int gareaid, void *local_base, int size_byte)
{
    MPI_Aint base_toshare;

    validata_gareaid(gareaid, "garea_attach");

    MPI_Get_address(local_base, &base_toshare);

    MPI_Win_attach(win_garea_table[gareaid], local_base, size_byte);
    garea_base_addresses[gareaid] = base_toshare;

    MPI_Win_flush_all(win_garea_table[gareaid]);
    MPI_Win_flush_all(win_garea_base_addresses);
}

void garea_detach(int gareaid, void *base)
{
    validata_gareaid(gareaid, "garea_detach");

    MPI_Win_detach(win_garea_table[gareaid], base);
    garea_base_addresses[gareaid] = GAREA_BASE_ADDRESSES_INITIAL;

    MPI_Win_flush_all(win_garea_table[gareaid]);
    MPI_Win_flush_all(win_garea_base_addresses);
}

void garea_write(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr, int type)
{
    size_t data_element_remain, element_toget;
    MPI_Datatype mpitype;
    MPI_Aint disp, target_base_address;
    char *addr;
    int unitsize;

    validata_gareaid(gareaid, "garea_write");

    // Polling until garea is attached
    while (1)
    {
        target_base_address = fetch_remote_base_address(gareaid, target_world_rank);
        if (garea_is_attached_memory(target_base_address))
            break;
    }

    mpitype = decide_MPI_datatype(type);
    unitsize = decide_unitsize(type);
    data_element_remain = data_element;
    addr = src_addr;
    disp = target_base_address + offset;

    while (data_element_remain > 0)
    {
        if ((data_element_remain * unitsize) > MAX_TRANSFER_SIZE)
        {
            element_toget = MAX_TRANSFER_SIZE / unitsize;
        }
        else
        {
            element_toget = data_element_remain;
        }
        MPI_Put(addr, element_toget, mpitype, target_world_rank, disp, element_toget, mpitype, win_garea_table[gareaid]);
        data_element_remain -= element_toget;
        addr += element_toget * unitsize;
        disp += element_toget;
    }

    MPI_Win_flush(target_world_rank, win_garea_table[gareaid]);
}

void garea_read(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *dest_addr, int type)
{
    size_t data_element_remain, element_toget;
    MPI_Datatype mpitype;
    MPI_Aint disp, target_base_address;
    int unitsize;
    char *addr;

    validata_gareaid(gareaid, "garea_read");

    // Polling until garea is attached
    while (1)
    {
        target_base_address = fetch_remote_base_address(gareaid, target_world_rank);
        if (garea_is_attached_memory(target_base_address))
            break;
    }

    mpitype = decide_MPI_datatype(type);
    unitsize = decide_unitsize(type);
    data_element_remain = data_element;
    addr = dest_addr;
    disp = target_base_address + offset;

    while (data_element_remain > 0)
    {
        if ((data_element_remain * unitsize) > MAX_TRANSFER_SIZE)
        {
            element_toget = MAX_TRANSFER_SIZE / unitsize;
        }
        else
        {
            element_toget = data_element_remain;
        }
        MPI_Get(addr, element_toget, mpitype, target_world_rank, disp, element_toget, mpitype, win_garea_table[gareaid]);
        data_element_remain -= element_toget;
        addr += element_toget * unitsize;
        disp += element_toget;
    }

    MPI_Win_flush(target_world_rank, win_garea_table[gareaid]);
}

void garea_delete()
{
    int i;

    if (win_garea_base_addresses == MPI_WIN_NULL)
    {
        printf("error: %d win_garea_base_addresses is MPI_WIN_NULL\n", world_myrank);
        return;
    }

    MPI_Win_unlock_all(win_garea_base_addresses);
    MPI_Win_free(&win_garea_base_addresses);

    for (i = 0; i < gareaidctr; i++)
    {
        int gareaid = i;
        if (win_garea_table[gareaid] == MPI_WIN_NULL)
        {
            printf("error: %d win_garea_table is MPI_WIN_NULL\n", world_myrank);
            return;
        }

        MPI_Win_unlock_all(win_garea_table[gareaid]);
        MPI_Win_free(&win_garea_table[gareaid]);
    }

    free_common_tables();
}

void setup_common_tables(int num_gareas)
{

    initialize_garea_base_addresses(num_gareas);
    win_garea_table = (MPI_Win *)malloc(num_gareas * sizeof(MPI_Win));
}
void free_common_tables()
{
    free((void *)garea_base_addresses);
}

void initialize_garea_base_addresses(int num_gareas)
{
    int i;
    garea_base_addresses_byte = num_gareas * sizeof(MPI_Aint);
    garea_base_addresses = (MPI_Aint *)malloc(garea_base_addresses_byte);
    for (i = 0; i < num_gareas; i++)
        garea_base_addresses[i] = (MPI_Aint)GAREA_BASE_ADDRESSES_INITIAL;
    MPI_Win_create((void *)garea_base_addresses, world_nprocs, sizeof(MPI_Aint), MPI_INFO_NULL, MPI_COMM_WORLD, &win_garea_base_addresses);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_garea_base_addresses);
}

void initialize_garea(int gareaidctr)
{
    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &(win_garea_table[gareaidctr]));
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_garea_table[gareaidctr]);
}

bool garea_is_attached_memory(MPI_Aint target_base_address)
{

    if (target_base_address == GAREA_BASE_ADDRESSES_INITIAL)
    {
        return false;
    }

    return true;
}

MPI_Datatype decide_MPI_datatype(int type)
{
    switch (type)
    {
    case DATA_INT:
        return MPI_INT;
    case DATA_REAL4:
        return MPI_FLOAT;
    case DATA_REAL8:
        return MPI_DOUBLE;
    default:
        fprintf(stderr, "%d : garea() : ERROR : wrong data type\n", world_myrank);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}

int decide_unitsize(int type)
{
    switch (type)
    {
    case DATA_INT:
        return sizeof(int);
    case DATA_REAL4:
        return sizeof(float);
    case DATA_REAL8:
        return sizeof(double);
    default:
        fprintf(stderr, "%d : writearea() : ERROR : wrong data type\n", world_myrank);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}

MPI_Aint fetch_remote_base_address(int gareaid, int target_world_rank)
{
    MPI_Aint remote_base_address;
    size_t offset_tofetch_remote_base_address;
    offset_tofetch_remote_base_address = gareaid;

    MPI_Get(&remote_base_address, GAREA_BASE_ADDRESSES_ELEMENT_PER_WIN, MPI_AINT, target_world_rank, offset_tofetch_remote_base_address, GAREA_BASE_ADDRESSES_ELEMENT_PER_WIN, MPI_AINT, win_garea_base_addresses);
    MPI_Win_flush(target_world_rank, win_garea_base_addresses);

    return remote_base_address;
}

void validata_gareaid(int gareaid, char *function_name)
{
    if (gareaid >= gareaidctr)
    {
        fprintf(stderr, "%d : %s : ERROR : wrong gareaid\n", world_myrank);
        fprintf(stderr, "gareaidは%dより小さい値を入力してください\ns", gareaidctr);
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
}

// function to debug
void show_base_addresses()
{
    for (int i = 0; i < DEF_MAXNUM_GAREAS; i++)
    {
        printf("world_myrank %d: base_address[%d]=%lx\n", world_myrank, i, (long)garea_base_addresses[i]);
    }
}