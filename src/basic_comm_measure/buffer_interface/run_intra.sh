#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:15:00
#PJM -j

module purge
module load gcc hpcx cuda
unset OMPI_MCA_rmaps_ppr_n_pernode
export UCX_IB_GPU_DIRECT_RDMA=1 
export UCX_TLS=rc,cuda_copy

vis_x_split=8
vis_z_split=8
vis_ppr=64

sim_ppr=4
node=1
sim_procs=$(($sim_ppr*$node))
sim_z_split=$(($sim_procs))
vis_procs=$(($vis_ppr*$node))
all_procs=$(($sim_procs+$vis_procs+1))
first_slots=$(($sim_ppr+$vis_ppr+1))

# 計算と比較
if [ $((vis_ppr * node)) -ne $((vis_x_split * vis_z_split)) ]; then
    echo "Error: vis_ppr * node と vis_x_split * vis_z_split が一致しません。"
    echo "vis_ppr * node = $((vis_ppr * node))"
    echo "vis_x_split * vis_z_split = $((vis_x_split * vis_z_split))"
    exit 1  # 強制終了
fi

echo "sim_procs = $sim_procs"
echo "vis_procs = $vis_procs"
echo "all_procs = $all_procs"
echo "sim_ppr = $sim_ppr"
echo "vis_ppr = $vis_ppr"
echo "first_slots = $first_slots"

echo > hostfile
NODELIST=(`cat ${PJM_O_NODEINF}`)
echo ${NODELIST[0]} slots=$first_slots > hostfile

#num=1は実行出来なかった
for num in 512
do
mpiexec -N $all_procs -mca plm_rsh_agent /bin/pjrsh -machinefile hostfile -mca btl_openib_allow_ib true \
-np $sim_ppr ./buffer_write_req $num : -np 1 ./buffer_write_cpl : -np $vis_ppr ./buffer_write_wrk $vis_x_split $vis_z_split $vis_procs \
> ./log/num${num}_node${node}_simprocs${sim_procs}_visprocs${vis_procs}.log
done 
