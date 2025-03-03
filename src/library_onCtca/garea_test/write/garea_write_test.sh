#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:01:00
#PJM -j

module purge
module load gcc hpcx cuda
unset OMPI_MCA_rmaps_ppr_n_pernode
export OMP_NUM_THREADS=1
export UCX_IB_GPU_DIRECT_RDMA=1
export UCX_TLS=rc,gdr_copy,cuda_copy

echo `head -n 1 ${PJM_O_NODEINF}` slots=1 > hosts1

head -n 1 ${PJM_O_NODEINF} > hosts2

echo `head -n 1 ${PJM_O_NODEINF}` slots=1 > hosts31

# mpiexec --oversubscribe -n 4 --hostfile hosts1 ./requester_garea_write : -n 1 --hostfile hosts2 ./coupler_garea_write : -n 4 --hostfile hosts31 ./worker1_garea_write;
mpiexec --oversubscribe -n 4 --hostfile hosts1 ./requester_garea_write_D : -n 1 --hostfile hosts2 ./coupler_garea_write : -n 4 --hostfile hosts31 ./worker1_garea_write;
