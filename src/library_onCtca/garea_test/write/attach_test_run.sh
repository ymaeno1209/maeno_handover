#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:01:00
#PJM -j
module load gcc hpcx cuda
unset OMPI_MCA_rmaps_ppr_n_pernode
export OMP_NUM_THREADS=1
export UCX_IB_GPU_DIRECT_RDMA=1
export UCX_TLS=rc,cuda_copy,gdr_copy

# mpiexec --oversubscribe -np 3 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:3:node ./attach_test0
mpiexec --oversubscribe -np 3 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:3:node ./attach_test

