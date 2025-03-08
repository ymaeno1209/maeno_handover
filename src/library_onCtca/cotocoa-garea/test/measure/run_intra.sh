#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM -j

module load gcc hpcx cuda
export OMP_NUM_THREADS=1
unset OMPI_MCA_rmaps_ppr_n_pernode

rm intra*.log 

mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./put > intra_put_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./garea_write > intra_garea_write_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./get > intra_get_int.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./garea_read > intra_garea_read_int.log
