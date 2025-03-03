#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:15:00
#PJM -j

module purge
module load gcc hpcx cuda
export OMP_NUM_THREADS=1
export UCX_IB_GPU_DIRECT_RDMA=1
export UCX_TLS=rc,cuda_copy
unset OMPI_MCA_rmaps_ppr_n_pernode

#num1=32までは検証可能。しかし32だと収束が速いため画像で正常かの判断が困難

step_num=2
vis_ppr=4
vis_x_split=2
vis_z_split=2
sim_times_per_step=100

node=1
sim_ppr=4
sim_procs=$(($sim_ppr*$node))
sim_z_split=$sim_procs
vis_procs=$(($vis_ppr*$node))
all_procs=$(($sim_procs+$vis_procs+1))
first_slots=$(($sim_ppr+$vis_ppr+1))
other_slots=$(($sim_ppr+$vis_ppr))

# 計算と比較
if [ $((vis_ppr * node)) -ne $((vis_x_split * vis_z_split)) ]; then
    echo "Error: vis_ppr * node と vis_x_split * vis_z_split が一致しません。"
    echo "vis_ppr * node = $((vis_ppr * node))"
    echo "vis_x_split * vis_z_split = $((vis_x_split * vis_z_split))"
    exit 1  # 強制終了
fi

echo "node=$node"
echo "sim_ppr = $sim_ppr"
echo "sim_procs = $sim_procs"
echo "vis_procs = $vis_procs"
echo "all_procs = $all_procs"
echo "vis_ppr = $vis_ppr"
echo "first_slots = $first_slots"
echo "other_slots = $other_slots"
echo "step_num = $step_num"
echo "sim_times_per_step = $sim_times_per_step"

echo > hostfile

NODELIST=(`cat ${PJM_O_NODEINF}`)

echo ${NODELIST[0]} slots=$first_slots > hostfile

mkdir -p log

for num in 256
do
echo "num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step} program start"
rm -rf num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step}
mkdir num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step}
for i in `seq 0 $((step_num-1))`
do 
    mkdir num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step}/step${i}
done
mpiexec -N $all_procs -mca plm_rsh_agent /bin/pjrsh -machinefile hostfile -mca btl_openib_allow_ib true \
-np $sim_ppr ./diff_requester $num $step_num $sim_times_per_step : -np 1 ./diff_coupler $num $step_num : -np $vis_ppr ./diff_worker $num $step_num $vis_x_split $vis_z_split $vis_procs $node $sim_ppr $vis_ppr $sim_times_per_step\
> ./log/num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step}.log
echo "num${num}_node${node}_simppr${sim_ppr}_visppr${vis_ppr}_stepnum${step_num}_calctimes${sim_times_per_step} program end"
done


