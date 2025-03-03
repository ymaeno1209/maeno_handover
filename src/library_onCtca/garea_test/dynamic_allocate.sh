#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:01:00
#PJM -j
module load gcc hpcx cuda
export OMP_NUM_THREADS=1
export UCX_IB_GPU_DIRECT_RDMA=0
export UCX_TLS=rc,cuda_copy,gdr_copy

# mpiexec -np 3 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:3:node  ./dynamic_allocate
mpiexec -np 3 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:3:node  ./dynamic_allocate_D //まだエラーが起こる