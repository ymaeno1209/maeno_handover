extern void call_stencil(int nprocs, int rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t *stream);
extern void call_top_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t top_stream);
extern void call_bot_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t bot_stream);
extern void call_mid_stencil(int calc, int new_nprocs, int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float dt, float kappa, float *f, float *fn, cudaStream_t mid_stream);
extern void call_copy_data(float *dest, float *src, int size);
extern void call_kernel_init(int new_rank, int nx, int ny, int nz, int mgn, float dx, float dy, float dz, float *f);
extern void call_packing_data(float *dest, float *src, int copy_element, int src_start_element, int dest_stride_element, int src_stride_element, int loop, cudaStream_t packing_stream);
extern void call_con_packing_data(float *dest_data, float *src_data, int num, cudaStream_t con_packing_stream);