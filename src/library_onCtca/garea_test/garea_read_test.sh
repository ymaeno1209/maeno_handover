#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:01:00
#PJM -j
module load gcc hpcx cuda
unset OMPI_MCA_rmaps_ppr_n_pernode

echo `head -n 1 ${PJM_O_NODEINF}` slots=1 > hosts1

head -n 1 ${PJM_O_NODEINF} > hosts2

echo `head -n 1 ${PJM_O_NODEINF}` slots=1 > hosts31

mpiexec --oversubscribe -n 1 --hostfile hosts1 ./requester_garea_read : -n 1 --hostfile hosts2 ./coupler_garea_read : -n 1 --hostfile hosts31 ./worker1_garea_read;
