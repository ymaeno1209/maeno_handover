#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:01:00
#PJM -j

module purge
module load gcc hpcx cuda

unset OMPI_MCA_rmaps_ppr_n_pernode
export UCX_IB_GPU_DIRECT_RDMA=1 
export UCX_TLS=rc,cuda_copy

echo > hostfile
NODELIST=(`cat ${PJM_O_NODEINF}`)
echo ${NODELIST[0]} slots=3 > hostfile

mpiexec -N 3 -mca plm_rsh_agent /bin/pjrsh -machinefile hostfile -mca btl_openib_allow_ib true \
-np 1 ./test_req : -np 1 ./test_cpl : -np 1 ./test_wrk