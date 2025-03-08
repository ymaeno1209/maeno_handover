#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=2
#PJM -L elapse=00:10:00
#PJM -j

module load gcc hpcx cuda
export OMP_NUM_THREADS=1
unset OMPI_MCA_rmaps_ppr_n_pernode

rm inter*.log 

mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:1:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./put > inter_put_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:1:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./garea_write > inter_garea_write_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:1:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./get > inter_get_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:1:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./garea_read > inter_garea_read_int.log
