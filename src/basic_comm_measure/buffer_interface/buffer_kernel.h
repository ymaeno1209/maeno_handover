#include <cuda_runtime.h>

extern void
call_pack_data(char *dest_data, char *src_data, size_t src_data_offset_start, size_t packed_data_byte, size_t width_byte, size_t square_byte, cudaStream_t packing_stream);

extern void
call_con_pack_data(char *dest_data, char *src_data, size_t *src_data_offset_start, size_t *packed_data_byte, size_t packed_data_byte_max, size_t *width_byte, size_t square_byte, int con_packing_count, cudaStream_t packing_stream);