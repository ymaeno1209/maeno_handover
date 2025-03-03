#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM -j

module purge
module load gcc hpcx cuda

rm intra*.log

#put
#rank0がputする
#./osu_put_latency rank0 rank1
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node  -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy ./garea_write -d cuda D H  > intra_gdr1_garea_write_g2c.log
mpiexec -np 2 -mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} --map-by ppr:2:node  -x UCX_IB_GPU_DIRECT_RDMA=0 -x UCX_TLS=rc,cuda_copy ./garea_write -d cuda D H  > intra_gdr0_garea_write_g2c.log