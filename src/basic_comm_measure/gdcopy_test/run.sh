#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM -j

module load gcc hpcx cuda
export OMP_NUM_THREADS=1
#export UCX_IB_GPU_DIRECT_RDMA=1
#export UCX_TLS=rc,gdr_copy,cuda_copy
# export UCX_TLS=rc,cuda_copy
unset OMPI_MCA_rmaps_ppr_n_pernode

# mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node  ./put_latency_test

# mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node  ./put_latency_g2c_test

mpiexec -np 1 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:1:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./gpu_memory_win_create