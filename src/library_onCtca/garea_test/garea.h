#include <mpi.h>
#include <stdbool.h>

void CTCAR_garea_create(int *gareaid);
void CTCAC_garea_create(int *gareaid);
void CTCAW_garea_create(int *gareaid);

void CTCAR_garea_attach(int gareaid, void *base,int size_byte);
void CTCAC_garea_attach(int gareaid, void *base,int size_byte);
void CTCAW_garea_attach(int gareaid, void *base,int size_byte);

void CTCAR_garea_detach(int gareaid, void *base);
void CTCAC_garea_detach(int gareaid, void *base);
void CTCAW_garea_detach(int gareaid, void *base);

void CTCAR_garea_delete();
void CTCAC_garea_delete();
void CTCAW_garea_delete();

void CTCAR_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAR_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAR_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAC_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAC_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAC_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAW_garea_read_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAW_garea_read_real4(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);
void CTCAW_garea_read_real8(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr);

void CTCAR_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *src_addr);
void CTCAC_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *src_addr);
void CTCAW_garea_write_int(int gareaid, int target_world_rank, size_t offset, size_t size, void *src_addr);
void CTCAR_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);
void CTCAC_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);
void CTCAW_garea_write_real4(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);
void CTCAR_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);
void CTCAC_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);
void CTCAW_garea_write_real8(int gareaid, int target_world_rank, size_t offset, size_t data_element, void *src_addr);

void garea_create(int num_gareas, int* gareaid);
void garea_attach(int gareaid, void *base,int size_byte);
void garea_read(int gareaid, int target_world_rank, size_t offset, size_t size, void *dest_addr, int type);
void garea_write(int gareaid, int target_world_rank, size_t offset, size_t size, void *src_addr, int type);
void garea_detach(int gareaid, void *base);
void garea_delete();

bool garea_is_attached_memory(MPI_Aint target_base_address);
int decide_unitsize(int type);
MPI_Datatype decide_MPI_datatype(int type);
MPI_Aint fetch_remote_base_address(int gareaid, int target_world_rank);
void setup_common_tables(int num_gareas);
void initialize_garea_base_addresses(int num_gareas);
void initialize_garea(int gareaidctr);
void free_common_tables();
void validata_gareaid(int gareaid, char* function_name);

//function to debug
void show_base_addresses();
