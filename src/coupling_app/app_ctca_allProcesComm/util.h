float output_float_time(struct timespec start_time, struct timespec end_time);
void calc_measure_time(float *elapsed_time, int iteration, float *average_time, float *max_time, float *min_time);
float reduce_max_gtime(float *ltime_list, float *max_gtime, int step_num, MPI_Comm new_comm);
